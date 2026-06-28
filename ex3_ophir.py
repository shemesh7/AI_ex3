import math
import heapq
from collections import deque

import ext_elev

# Put your real submission id(s) here before handing in.
id = ["000000000"]

INF = float("inf")


class _PlanAbort(Exception):
    """Raised when the bounded planner exceeds its node budget -> fall back."""


class Controller:
    """RL controller for the stochastic Multi-Elevator MDP, with a general
    bounded-lookahead planning fallback layered on top of a fast heuristic.

    The transition model is HIDDEN: elevator/person success probabilities and
    per-person delivery rewards are unknown and learned online (model-based
    active TD) from observed transitions and api.get_last_gained_reward().

    Default policy (fast, unchanged from the heuristic controller)
    -------------------------------------------------------------
    1. BFS distance-to-goal per person (capacity-filtered graph).
    2. Observer: estimate P-hat (MOVE/ENTER/EXIT success) and R-hat (per-person
       reward) by diffing prev_state -> state every step.
    3. Canonical UCB1 on R-hat for exploration.
    4. phi: max-product (model-based EV) routing potential per person.
    5. Soft multi-agent claiming + a reliability gate (_should_serve).
    6. Greedy priority + EV action selection, with productive transfer hand-offs
       and same-direction detour carpooling.

    Planning fallback (NEW, general)
    --------------------------------
    A bounded expectimax over the *learned* model (P-hat / R-hat) refines the
    greedy choice ONLY when the current state is small enough to plan cheaply
    AND the greedy decision looks non-obvious (several actions with nearly equal
    heuristic value). It expands only the heuristic's top candidate actions, uses
    memoisation and a hard node budget, and OVERRIDES the greedy action only when
    its expected look-ahead value beats the greedy action's by a margin. Because
    it never overrides without a strict expected-value improvement, it cannot do
    worse than the heuristic in expectation, and it falls back to the heuristic
    whenever planning would be too expensive.

    The triggers are deliberately STRUCTURAL (few remaining persons -> small
    sub-MDP; near-tie heuristic scores -> greedy ambiguity). They are read from
    the live state and the learned model only -- never from problem identity,
    seeds, or any benchmark-specific counts/constants.

    Nothing persists between instances: every estimate is initialised in
    __init__, so each seed/problem starts from a clean slate.
    """

    # ---- heuristic constants (unchanged) ---------------------------------- #
    GAMMA = 0.99         # per-hop discount (finite-horizon appropriate)
    UCB_C = 2.0          # UCB1 confidence scale C (bounded; NOT the max reward)
    OPTIMISTIC = 100.0   # optimistic init: deliver each person once before exploiting
    STICKY = 1.0         # incumbency bonus that damps claim thrashing
    CLAIM_PENALTY = 0.15 # soft-claim score multiplier when another elevator owns a target
    FARM_MARGIN = 1.05   # how much better the farm rate must be before we reset-loop

    # ---- planning-fallback hyperparameters (general, not per-instance) ----- #
    # These bound the *cost* of the look-ahead and the *confidence* needed to
    # override the heuristic. They are problem-agnostic: they describe how much
    # compute we are willing to spend and how sure we must be, not anything
    # about a particular map, reward table, or horizon.
    PLAN_MAX_PERSONS = 3     # only plan when this few persons remain (tractable sub-MDP)
    PLAN_DEPTH = 5           # bounded action look-ahead depth
    PLAN_BRANCH = 4          # max candidate actions expanded per node
    PLAN_NODE_BUDGET = 6000  # hard cap on simulated nodes; exceed -> fall back
    PLAN_CLOSE_EPS = 0.10    # "scores are close" relative threshold (greedy ambiguous)
    PLAN_OVERRIDE_MARGIN = 0.02  # only override greedy on a clear EV improvement
    PLAN_GAMMA = 0.99        # discount used inside the look-ahead
    FARM_SUBSET_MAX = 6      # enumerate farm subsets only when persons are this few

    def __init__(self, game: ext_elev.GameAPI):
        self.game = game

        # ---- static configuration (allowed by the API) ------------------- #
        self.reachable = {e: set(fs) for e, fs in game.get_reachable().items()}
        self.capacities = dict(game.get_capacities())
        self.goal_reward = float(game.get_goal_reward())

        init_state = game.get_initial_state()
        self.initial_state = init_state
        self.person_ids = [pid for pid, _ in init_state[1]]
        self.person_weight = {p: game.get_person_weight(p) for p in self.person_ids}
        self.person_goal = {p: game.get_person_goal(p) for p in self.person_ids}
        self.person_start = {
            pid: loc[1] for pid, loc in init_state[1] if loc[0] == "floor"
        }

        # ---- (1) BFS distance-to-goal per person ------------------------- #
        self.dist = {p: self._bfs_to_goal(p) for p in self.person_ids}

        # ---- (2) learned statistics (the Observer) ----------------------- #
        self.move_succ = {e: 0 for e in self.reachable}
        self.move_fail = {e: 0 for e in self.reachable}
        self.enter_succ = {p: 0 for p in self.person_ids}
        self.enter_fail = {p: 0 for p in self.person_ids}
        self.exit_succ = {p: 0 for p in self.person_ids}
        self.exit_fail = {p: 0 for p in self.person_ids}

        # ---- (3) UCB1 reward statistics ---------------------------------- #
        self.deliver_sum = {p: 0.0 for p in self.person_ids}
        self.deliver_count = {p: 0 for p in self.person_ids}
        self.total_deliveries = 0

        # ---- (5) multi-agent coordination -------------------------------- #
        self.claimed_targets = {}

        # ---- Observer memory: previous (state, action) ------------------- #
        self.prev_state = None
        self.prev_action = None

        # planner bookkeeping (reset per call)
        self._plan_nodes = 0

    # ====================================================================== #
    # (1) BFS pre-computation
    # ====================================================================== #
    def _bfs_to_goal(self, p):
        """Unweighted BFS from p's goal over the capacity-filtered graph."""
        weight = self.person_weight[p]
        goal = self.person_goal[p]
        allowed = [e for e in self.reachable if weight <= self.capacities[e]]

        dist = {goal: 0}
        queue = deque([goal])
        while queue:
            f = queue.popleft()
            for e in allowed:
                if f in self.reachable[e]:
                    for nxt in self.reachable[e]:
                        if nxt not in dist:
                            dist[nxt] = dist[f] + 1
                            queue.append(nxt)
        return dist

    def _reachable_for(self, p, floor):
        return floor in self.dist[p]

    # ====================================================================== #
    # (2) The Observer: learn the hidden model from the last transition
    # ====================================================================== #
    def _observe(self, prev_state, action, cur_state):
        kind, a, b = self._parse(action)
        if kind is None:  # RESET / unparsable: nothing to learn
            return

        prev_persons = dict(prev_state[1])
        cur_persons = dict(cur_state[1])
        prev_elevs = {e: f for e, f, _ in prev_state[0]}
        cur_elevs = {e: f for e, f, _ in cur_state[0]}

        if kind == "MOVE":
            e_id, target = a, b
            prev_f = prev_elevs.get(e_id)
            new_f = cur_elevs.get(e_id)
            if prev_f is not None and target != prev_f:
                if new_f == target:
                    self.move_succ[e_id] += 1
                else:
                    self.move_fail[e_id] += 1

        elif kind == "ENTER":
            p_id, e_id = a, b
            if cur_persons.get(p_id) == ("in", e_id):
                self.enter_succ[p_id] += 1
            else:
                self.enter_fail[p_id] += 1

        elif kind == "EXIT":
            p_id, e_id = a, b
            was = prev_persons.get(p_id)
            still_inside = (cur_persons.get(p_id) == ("in", e_id))
            if still_inside:
                self.exit_fail[p_id] += 1
            else:
                self.exit_succ[p_id] += 1
                exited_at_goal = (
                    was == ("in", e_id)
                    and prev_elevs.get(e_id) == self.person_goal[p_id]
                )
                if exited_at_goal:
                    gained = float(self.game.get_last_gained_reward())
                    # The engine bundles goal_reward into the final delivery's
                    # step; strip it so we learn the person's own reward only.
                    if prev_state[2] == 1:
                        gained -= self.goal_reward
                    self.deliver_sum[p_id] += gained
                    self.deliver_count[p_id] += 1
                    self.total_deliveries += 1

    @staticmethod
    def _parse(action):
        if action is None or action.startswith("RESET"):
            return None, None, None
        kind, rest = action.split("{", 1)
        x, y = rest.rstrip("}").split(",")
        return kind, int(x), int(y)

    # ====================================================================== #
    # (3) UCB1 reward estimate + R-hat / P-hat models
    # ====================================================================== #
    def _ucb_value(self, p):
        """UCB1 upper confidence bound on person p's delivery reward (R-hat)."""
        n = self.deliver_count[p]
        if n == 0:
            return self.OPTIMISTIC
        mean = self.deliver_sum[p] / n
        t = max(self.total_deliveries, 1)
        bonus = self.UCB_C * math.sqrt(2.0 * math.log(t) / n)
        return mean + bonus

    def _reward_estimate(self, p):
        """Best point estimate of p's delivery reward (R-hat mean, no optimism).

        Used by the *planner* for value estimation -- the UCB bonus is an
        exploration device, but value look-ahead wants the expected reward.
        Unvisited persons fall back to the average observed reward, then to a
        neutral prior, so the estimate is always well defined and bounded.
        """
        if self.deliver_count[p] > 0:
            return self.deliver_sum[p] / self.deliver_count[p]
        if self.total_deliveries > 0:
            return sum(self.deliver_sum.values()) / self.total_deliveries
        return 1.0

    def _p_move(self, e):
        """P-hat: estimated MOVE success probability of elevator e (optimistic init)."""
        s, f = self.move_succ[e], self.move_fail[e]
        return (s + 1.0) / (s + f + 1.0)

    def _p_enter(self, p):
        """P-hat: estimated ENTER success probability of person p."""
        s, f = self.enter_succ[p], self.enter_fail[p]
        return (s + 1.0) / (s + f + 2.0)

    def _p_exit(self, p):
        """P-hat: estimated EXIT success probability of person p."""
        s, f = self.exit_succ[p], self.exit_fail[p]
        return (s + 1.0) / (s + f + 2.0)

    # ====================================================================== #
    # (4) Probabilistic routing potential phi(p, floor)
    # ====================================================================== #
    def _compute_phi(self, p):
        """Max-product shortest path from p's goal (model-based EV multiplier).

        phi is state-independent given the current P-hat estimates (it depends
        only on the graph and learned reliabilities), so the planner computes it
        ONCE per decision and reuses it across every simulated state.
        """
        goal = self.person_goal[p]
        weight = self.person_weight[p]
        allowed = [e for e in self.reachable if weight <= self.capacities[e]]
        rel = {e: self.GAMMA * self._p_move(e) for e in allowed}

        phi = {goal: 1.0}
        heap = [(-1.0, goal)]
        while heap:
            negv, f = heapq.heappop(heap)
            v = -negv
            if v < phi.get(f, 0.0) - 1e-12:
                continue
            for e in allowed:
                fs = self.reachable[e]
                if f not in fs:
                    continue
                cand = v * rel[e]
                for f2 in fs:
                    if f2 != f and cand > phi.get(f2, 0.0) + 1e-12:
                        phi[f2] = cand
                        heapq.heappush(heap, (-cand, f2))
        return phi

    @staticmethod
    def _best_ride_target(reachable_floors, phi, cur_f):
        best_f, best_pot = cur_f, phi.get(cur_f, 0.0)
        for f2 in reachable_floors:
            pv = phi.get(f2, 0.0)
            if pv > best_pot:
                best_pot, best_f = pv, f2
        return best_f, best_pot

    # ====================================================================== #
    # (5) Soft multi-agent coordination + reliability gate
    # ====================================================================== #
    def _assign_claims(self, floor_of, elevators, busy, phi_map):
        pairs = []
        for p, fp in floor_of.items():
            if not self._reachable_for(p, fp):
                continue
            phi = phi_map[p]
            w = self.person_weight[p]
            base = self._ucb_value(p) * self._p_enter(p)
            for eid, cur_f, cur_w in elevators:
                if busy.get(eid):
                    continue
                if cur_w + w > self.capacities[eid]:
                    continue
                if fp not in self.reachable[eid]:
                    continue
                if not self._should_serve(eid, p, fp, phi):
                    continue
                _, tpot = self._best_ride_target(self.reachable[eid], phi, fp)
                if tpot <= phi.get(fp, 0.0) + 1e-12:
                    continue
                travel = 1.0 if cur_f == fp else self.GAMMA * self._p_move(eid)
                score = base * travel * self.GAMMA * self._p_move(eid) * tpot
                if self.claimed_targets.get(p) == eid:
                    score *= self.STICKY
                pairs.append((score, p, eid))

        pairs.sort(key=lambda t: t[0], reverse=True)
        claims, used = {}, set()
        for _, p, eid in pairs:
            if p in claims or eid in used:
                continue
            claims[p] = eid
            used.add(eid)
        self.claimed_targets = claims
        return claims

    def _should_serve(self, eid, p, floor, phi):
        """Reliability gate: don't commit a person to a much-less-reliable lift
        when a more reliable one can advance them (no parallelism -> extra
        retries are pure waste). Route-aware: `best` ranges only over lifts that
        can actually advance p from `floor`, so a mandatory unreliable transfer
        is still allowed and no one is starved."""
        pm = self._p_move(eid)
        w = self.person_weight[p]
        cur_pot = phi.get(floor, 0.0)
        best = 0.0
        for e in self.reachable:
            if w > self.capacities[e] or floor not in self.reachable[e]:
                continue
            _, tpot = self._best_ride_target(self.reachable[e], phi, floor)
            if tpot > cur_pot + 1e-12:
                r = self._p_move(e)
                if r > best:
                    best = r
        if best <= 0.0 or pm >= best - 1e-12:
            return True
        d = self.dist[p].get(floor, 1) or 1
        return d * (1.0 / pm - 1.0 / best) <= 1.0

    def _can_transfer(self, p, cur_f, exclude_e, phi):
        w = self.person_weight[p]
        cur_pot = phi.get(cur_f, 0.0)
        for e2 in self.reachable:
            if e2 == exclude_e or w > self.capacities[e2]:
                continue
            if cur_f not in self.reachable[e2]:
                continue
            _, tpot = self._best_ride_target(self.reachable[e2], phi, cur_f)
            if tpot > cur_pot + 1e-12:
                return True
        return False

    # ====================================================================== #
    # (6) Candidate generation (shared by the greedy policy and the planner)
    # ====================================================================== #
    def _candidate_actions(self, state, phi_map, claims=None):
        """Return [(priority, score, action_str)] for `state`.

        This is EXACTLY the heuristic's per-elevator candidate enumeration,
        factored out so both the greedy policy (take the best) and the planner
        (expand the top few) use one consistent definition of "sensible legal
        actions". `claims` only biases Priority-1 driving scores; the planner
        passes none (it explores without the coordination bias)."""
        claims = claims or {}
        elevators_t, persons_t, _ = state

        floor_of, onboard_of = {}, {}
        for pid, loc in persons_t:
            if loc[0] == "floor":
                floor_of[pid] = loc[1]
            else:
                onboard_of.setdefault(loc[1], []).append(pid)
        elevators = list(elevators_t)

        candidates = []
        for eid, cur_f, cur_w in elevators:
            onboard = onboard_of.get(eid, [])
            reach = self.reachable[eid]
            rel_e = self._p_move(eid)

            # Priority 3: drop a passenger at their goal.
            for p in onboard:
                if cur_f == self.person_goal[p]:
                    candidates.append(
                        (3, self._ucb_value(p) * self._p_exit(p), f"EXIT{{{p},{eid}}}")
                    )

            # Priority 1 (carry) / Priority 2 (productive transfer hand-off).
            best_move = None
            for p in onboard:
                if cur_f == self.person_goal[p]:
                    continue
                phi = phi_map[p]
                tgt, tpot = self._best_ride_target(reach, phi, cur_f)
                if tpot > phi.get(cur_f, 0.0) + 1e-12 and tgt != cur_f:
                    score = self._ucb_value(p) * self.GAMMA * rel_e * tpot
                    if best_move is None or score > best_move[0]:
                        best_move = (score, tgt)
                elif self._can_transfer(p, cur_f, eid, phi):
                    score = self._ucb_value(p) * self._p_exit(p) * phi.get(cur_f, 0.0)
                    candidates.append((2, score, f"EXIT{{{p},{eid}}}"))

            if best_move is not None:
                candidates.append((1, best_move[0], f"MOVE{{{eid},{best_move[1]}}}"))

            # Detour carpooling: collect a same-direction fare en route.
            for A in onboard:
                if cur_f == self.person_goal[A]:
                    continue
                phiA = phi_map[A]
                T_A, tA = self._best_ride_target(reach, phiA, cur_f)
                if tA <= phiA.get(cur_f, 0.0) + 1e-12:
                    continue
                for B, fB in floor_of.items():
                    if fB == cur_f or fB not in reach or fB == T_A:
                        continue
                    if cur_w + self.person_weight[B] > self.capacities[eid]:
                        continue
                    if any(phi_map[C].get(fB, 0.0) <= 0.0 for C in onboard):
                        continue
                    if not self._should_serve(eid, B, fB, phi_map[B]):
                        continue
                    phiB = phi_map[B]
                    if phiB.get(T_A, 0.0) <= phiB.get(fB, 0.0) + 1e-12:
                        continue
                    score = (self._ucb_value(A) * phiA.get(fB, 0.0)
                             + self._ucb_value(B) * phiB.get(T_A, 0.0)
                             ) * self.GAMMA * rel_e
                    candidates.append((1, score, f"MOVE{{{eid},{fB}}}"))

            # Priority 2: board a co-located waiting person who is advanced.
            for p, fp in floor_of.items():
                if fp != cur_f or cur_w + self.person_weight[p] > self.capacities[eid]:
                    continue
                if not self._should_serve(eid, p, cur_f, phi_map[p]):
                    continue
                phi = phi_map[p]
                _, tpot = self._best_ride_target(reach, phi, cur_f)
                if tpot > phi.get(cur_f, 0.0) + 1e-12:
                    score = (self._ucb_value(p) * self._p_enter(p)
                             * self.GAMMA * rel_e * tpot)
                    candidates.append((2, score, f"ENTER{{{p},{eid}}}"))

            # Priority 1 (empty): drive toward the best fare (soft claim bias).
            if not onboard:
                best_fetch = None
                for p, fp in floor_of.items():
                    if fp == cur_f or fp not in reach:
                        continue
                    if cur_w + self.person_weight[p] > self.capacities[eid]:
                        continue
                    if not self._should_serve(eid, p, fp, phi_map[p]):
                        continue
                    phi = phi_map[p]
                    _, tpot = self._best_ride_target(reach, phi, fp)
                    if tpot <= phi.get(fp, 0.0) + 1e-12:
                        continue
                    score = self._ucb_value(p) * self.GAMMA * rel_e * tpot
                    owner = claims.get(p)
                    if owner is not None and owner != eid:
                        score *= self.CLAIM_PENALTY
                    if best_fetch is None or score > best_fetch[0]:
                        best_fetch = (score, fp)
                if best_fetch is not None:
                    candidates.append((1, best_fetch[0], f"MOVE{{{eid},{best_fetch[1]}}}"))

        return candidates

    # ====================================================================== #
    # Action selection: heuristic default + planning fallback
    # ====================================================================== #
    def choose_next_action(self, state):
        if self.prev_state is not None:
            self._observe(self.prev_state, self.prev_action, state)

        action = self._decide(state)

        self.prev_state = state
        self.prev_action = action
        return action

    def _decide(self, state):
        elevators_t, persons_t, total_remaining = state

        # ---- (7) RESET-farming short-circuit (generalised) --------------- #
        farm_set = self._farm_set()
        if farm_set is not None:
            present = {pid for pid, _ in persons_t}
            if not (farm_set & present):       # every farmed person delivered
                if state != self.initial_state:
                    return "RESET"             # respawn them and repeat
            else:
                action = self._pursue_targets(state, farm_set)
                if action is not None:
                    return action
                if state != self.initial_state:
                    return "RESET"

        # ---- heuristic candidate set + greedy choice --------------------- #
        floor_of, onboard_of = {}, {}
        for pid, loc in persons_t:
            if loc[0] == "floor":
                floor_of[pid] = loc[1]
            else:
                onboard_of.setdefault(loc[1], []).append(pid)
        elevators = list(elevators_t)
        busy = {eid: bool(onboard_of.get(eid)) for eid, _, _ in elevators}

        # phi is state-independent (depends only on goals/weights/P-hat), so we
        # build it for EVERY person once -- the planner may simulate a completion
        # that resets to the initial state, where all persons are present again.
        phi_map = {pid: self._compute_phi(pid) for pid in self.person_ids}
        claims = self._assign_claims(floor_of, elevators, busy, phi_map)
        candidates = self._candidate_actions(state, phi_map, claims)

        if not candidates:
            if state != self.initial_state:
                return "RESET"
            return self._fallback(state)

        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        greedy_action = candidates[0][2]

        # ---- planning fallback (only when structurally warranted) -------- #
        if self._should_plan(total_remaining, candidates):
            planned = self._plan(state, phi_map, candidates, greedy_action)
            if planned is not None:
                return planned

        return greedy_action

    # ====================================================================== #
    # Planning fallback: bounded expectimax over the LEARNED model
    # ====================================================================== #
    def _should_plan(self, total_remaining, candidates):
        """STRUCTURAL trigger -- no problem identity is consulted.

        We plan only when BOTH hold:
          * the remaining sub-problem is small (total_remaining <= a tractable
            bound) so a bounded expectimax is cheap; every instance eventually
            enters this regime as people are delivered (endgame), and small
            instances live in it throughout; and
          * the greedy choice is genuinely ambiguous -- the top two candidate
            actions of the same priority have nearly equal heuristic scores --
            which is exactly the situation where one-step EV can pick the wrong
            one and a short look-ahead can disambiguate.

        Both conditions come purely from the current state and learned scores,
        so the trigger transfers unchanged if the map/rewards/horizon change.
        """
        if total_remaining > self.PLAN_MAX_PERSONS:
            return False
        if len(candidates) < 2:
            return False
        # candidates are not yet guaranteed sorted by caller in all paths; sort.
        top = sorted(candidates, key=lambda c: (c[0], c[1]), reverse=True)
        (p0, s0, _), (p1, s1, _) = top[0], top[1]
        if p0 != p1:               # different priority tiers -> greedy is decisive
            return False
        return (s0 - s1) <= self.PLAN_CLOSE_EPS * (abs(s0) + 1e-9)

    def _plan(self, state, phi_map, candidates, greedy_action):
        """Expand the heuristic's top candidates with bounded expectimax and
        return a better action, or None to keep the greedy choice.

        Safety: we evaluate the greedy action too and only override it when a
        sibling's expected look-ahead value clearly exceeds it -- so the planner
        is a strict improvement filter, never a gamble."""
        top = sorted(candidates, key=lambda c: (c[0], c[1]), reverse=True)
        actions = [a for _, _, a in top[:self.PLAN_BRANCH]]
        if greedy_action not in actions:
            actions.append(greedy_action)

        self._plan_nodes = 0
        memo = {}
        values = {}
        try:
            for a in actions:
                values[a] = self._q_value(state, a, self.PLAN_DEPTH, phi_map, memo)
        except _PlanAbort:
            return None  # too expensive this step -> trust the heuristic

        best_a = max(values, key=lambda a: values[a])
        g = values.get(greedy_action)
        if g is None:
            return None
        if best_a != greedy_action and values[best_a] > g * (1.0 + self.PLAN_OVERRIDE_MARGIN):
            return best_a
        return None

    def _state_value(self, state, depth, phi_map, memo):
        """Expectimax value of `state` with `depth` actions of look-ahead left."""
        persons_t = state[1]
        if not persons_t:
            return 0.0
        if depth <= 0:
            return self._leaf_value(state, phi_map)

        key = (state, depth)
        cached = memo.get(key)
        if cached is not None:
            return cached

        self._plan_nodes += 1
        if self._plan_nodes > self.PLAN_NODE_BUDGET:
            raise _PlanAbort()

        cands = self._candidate_actions(state, phi_map, {})
        if not cands:
            val = self._leaf_value(state, phi_map)
        else:
            cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
            val = -INF
            for _, _, a in cands[:self.PLAN_BRANCH]:
                qv = self._q_value(state, a, depth, phi_map, memo)
                if qv > val:
                    val = qv
            if val == -INF:
                val = self._leaf_value(state, phi_map)

        memo[key] = val
        return val

    def _q_value(self, state, action, depth, phi_map, memo):
        """Expected (immediate reward + discounted future value) of `action`,
        averaging over the learned stochastic outcomes (a chance node)."""
        total = 0.0
        for prob, reward, nxt in self._sim_outcomes(state, action):
            if prob <= 0.0:
                continue
            future = self._state_value(nxt, depth - 1, phi_map, memo)
            total += prob * (reward + self.PLAN_GAMMA * future)
        return total

    def _sim_outcomes(self, state, action):
        """Model the outcomes of `action` using ONLY learned estimates.

        Returns [(prob, immediate_reward, next_state)]. Reward uses R-hat means
        and the bundled goal_reward on the final delivery. A failed MOVE is
        approximated as "stay put" (a lost step) -- a deliberately simple,
        bounded 2-outcome model of the hidden failure dynamics that keeps the
        branching factor small; it is conservative (failure yields no progress)
        and matches the agent's reliability-aware bias."""
        kind, a, b = self._parse(action)
        elevators_t, persons_t, rem = state
        if kind is None:  # RESET
            return [(1.0, 0.0, self.initial_state)]

        elev = {e: (f, w) for e, f, w in elevators_t}

        if kind == "MOVE":
            e, target = a, b
            cf, cw = elev[e]
            q = self._p_move(e)
            succ = (self._with_elevator(elevators_t, e, target, cw), persons_t, rem)
            outs = [(q, 0.0, succ)]
            if q < 1.0:
                outs.append((1.0 - q, 0.0, state))
            return outs

        if kind == "ENTER":
            p, e = a, b
            cf, cw = elev[e]
            q = self._p_enter(p)
            nxt = (self._with_elevator(elevators_t, e, cf, cw + self.person_weight[p]),
                   self._with_person(persons_t, p, ("in", e)), rem)
            outs = [(q, 0.0, nxt)]
            if q < 1.0:
                outs.append((1.0 - q, 0.0, state))
            return outs

        if kind == "EXIT":
            p, e = a, b
            cf, cw = elev[e]
            q = self._p_exit(p)
            new_elevs = self._with_elevator(elevators_t, e, cf, cw - self.person_weight[p])
            if cf == self.person_goal[p]:                 # delivery
                reward = self._reward_estimate(p)
                new_persons = tuple(sorted(
                    (pid, loc) for pid, loc in persons_t if pid != p))
                new_rem = rem - 1
                if new_rem <= 0:                          # building complete
                    reward += self.goal_reward
                    succ = self.initial_state
                else:
                    succ = (new_elevs, new_persons, new_rem)
                outs = [(q, reward, succ)]
            else:                                         # drop off mid-route
                succ = (new_elevs, self._with_person(persons_t, p, ("floor", cf)), rem)
                outs = [(q, 0.0, succ)]
            if q < 1.0:
                outs.append((1.0 - q, 0.0, state))
            return outs

        return [(1.0, 0.0, state)]

    @staticmethod
    def _with_elevator(elevators_t, eid, floor, weight):
        return tuple(sorted(
            (eid, floor, weight) if e == eid else (e, f, w)
            for e, f, w in elevators_t))

    @staticmethod
    def _with_person(persons_t, pid, loc):
        return tuple(sorted(
            (pid, loc) if p == pid else (p, l) for p, l in persons_t))

    def _leaf_value(self, state, phi_map):
        """Heuristic value-to-go at the look-ahead horizon.

        Two model-based terms, both from learned estimates only:
          * expected still-collectable delivery reward, each person's R-hat
            scaled by phi at their current floor (closer & more reliable -> more
            of the reward is realistically reachable); and
          * a completion bonus: goal_reward weighted by how deliverable the whole
            remaining set is (product of per-person phi -> ~1 when all are at/near
            their goals), so the planner values *finishing* and will prefer
            completion when goal_reward is large relative to individual rewards.
        """
        elevators_t, persons_t, _ = state
        if not persons_t:
            return 0.0
        elev_floor = {e: f for e, f, _ in elevators_t}
        val = 0.0
        proximity = 1.0
        for p, loc in persons_t:
            floor = loc[1] if loc[0] == "floor" else elev_floor[loc[1]]
            ph = phi_map[p].get(floor, 0.0)
            val += self._reward_estimate(p) * ph
            proximity *= ph
        val += self.goal_reward * proximity
        return val

    # ====================================================================== #
    # (7) RESET-farming: general completion-vs-(solo/partial)-farming decision
    # ====================================================================== #
    def _solo_delivery_steps(self, p):
        """Expected actions to deliver p once from its start floor (no reset)."""
        hops = self.dist[p].get(self.person_start.get(p, None), INF)
        if hops == INF:
            return INF
        allowed = [e for e in self.reachable if self.person_weight[p] <= self.capacities[e]]
        if not allowed:
            return INF
        move_best = max(self._p_move(e) for e in allowed)
        per_hop = 1.0 / self._p_enter(p) + 1.0 / move_best + 1.0 / self._p_exit(p)
        return hops * per_hop

    def _farm_set(self):
        """Pick a SUBSET of persons to RESET-farm, or None to keep completing.

        Compares two stationary policies by expected reward-per-step, using only
        learned estimates:
          * completion: deliver everyone (the final delivery auto-resets and pays
            goal_reward), rate = (sum R-hat + goal_reward) / sum delivery_steps;
          * farm subset S: deliver only S, then RESET,
            rate = (sum_{S} R-hat) / (sum_{S} steps + 1).
        We farm only when the best subset rate beats completion by FARM_MARGIN.

        This is the general form of the solo-farming rule: it considers any
        subset (capped to singletons when there are too many persons to
        enumerate cheaply), and because goal_reward sits in the completion
        numerator, a large goal_reward makes completion win for *every* subset --
        so farming only triggers when a small, lucrative, cheap-to-repeat subset
        genuinely dominates finishing the building. No instance-specific
        thresholds are used; FARM_MARGIN is a general confidence hyperparameter.
        """
        if any(self.deliver_count[p] == 0 for p in self.person_ids):
            return None  # still exploring -- learn every reward first

        info = []
        total_reward = self.goal_reward
        total_steps = 0.0
        for p in self.person_ids:
            steps = self._solo_delivery_steps(p)
            if steps == INF:
                continue
            mean = self.deliver_sum[p] / self.deliver_count[p]
            info.append((p, mean, steps))
            total_reward += mean
            total_steps += steps
        if not info or total_steps <= 0.0:
            return None
        rate_complete = total_reward / total_steps

        best_set, best_rate = None, 0.0
        n = len(info)
        if n <= self.FARM_SUBSET_MAX:
            # enumerate every non-empty subset (n small -> cheap)
            for mask in range(1, 1 << n):
                r = s = 0.0
                for i in range(n):
                    if mask >> i & 1:
                        r += info[i][1]
                        s += info[i][2]
                rate = r / (s + 1.0)  # +1 for the RESET action
                if rate > best_rate:
                    best_rate = rate
                    best_set = frozenset(info[i][0] for i in range(n) if mask >> i & 1)
        else:
            # too many persons to enumerate subsets -> consider singletons only
            for p, mean, steps in info:
                rate = mean / (steps + 1.0)
                if rate > best_rate:
                    best_rate = rate
                    best_set = frozenset((p,))

        if best_set is None:
            return None
        return best_set if best_rate > rate_complete * self.FARM_MARGIN else None

    def _pursue_targets(self, state, targets):
        """Greedy delivery restricted to a set of target persons (farming mode)."""
        elevators_t, persons_t, _ = state
        loc_of = dict(persons_t)
        present = [t for t in targets if t in loc_of]
        if not present:
            return None

        candidates = []
        for target in present:
            loc = loc_of[target]
            phi = self._compute_phi(target)
            w = self.person_weight[target]
            for eid, cur_f, cur_w in elevators_t:
                reach = self.reachable[eid]
                rel_e = self._p_move(eid)
                if loc == ("in", eid):
                    if cur_f == self.person_goal[target]:
                        candidates.append((3, 1.0, f"EXIT{{{target},{eid}}}"))
                    else:
                        tgt, tpot = self._best_ride_target(reach, phi, cur_f)
                        if tpot > phi.get(cur_f, 0.0) + 1e-12 and tgt != cur_f:
                            candidates.append((1, tpot * rel_e, f"MOVE{{{eid},{tgt}}}"))
                elif loc[0] == "floor":
                    fp = loc[1]
                    if fp == cur_f:
                        if cur_w + w <= self.capacities[eid]:
                            _, tpot = self._best_ride_target(reach, phi, cur_f)
                            if tpot > phi.get(cur_f, 0.0) + 1e-12:
                                candidates.append((2, tpot * rel_e, f"ENTER{{{target},{eid}}}"))
                    elif cur_w == 0 and fp in reach and w <= self.capacities[eid]:
                        _, tpot = self._best_ride_target(reach, phi, fp)
                        if tpot > phi.get(fp, 0.0) + 1e-12:
                            candidates.append((1, tpot * rel_e, f"MOVE{{{eid},{fp}}}"))

        if candidates:
            candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
            return candidates[0][2]
        return None

    def _fallback(self, state):
        """Guarantee a legal action even when no heuristic candidate exists."""
        elevators_t, persons_t, _ = state
        floor_of = {pid: loc[1] for pid, loc in persons_t if loc[0] == "floor"}
        for eid, cur_f, cur_w in elevators_t:
            for p, f_p in floor_of.items():
                if f_p == cur_f and cur_w + self.person_weight[p] <= self.capacities[eid]:
                    return f"ENTER{{{p},{eid}}}"
        for eid, cur_f, _ in elevators_t:
            for f in self.reachable[eid]:
                if f != cur_f:
                    return f"MOVE{{{eid},{f}}}"
        return "RESET"
