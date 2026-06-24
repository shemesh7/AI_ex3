"""
ADP (Adaptive Dynamic Programming) + GLIE controller for the hidden-model
multi-elevator MDP.

Architecture
────────────
• Empirical model:  per-entity Bayesian counters for pe (elevator MOVE success)
  and qp (person ENTER/EXIT success); per-person running mean for Erew.
• Planner:          the full expectimax + A* engine from Assignment 2, fed the
  empirical estimates.  Plans are rebuilt whenever estimates change materially.
• Exploration:      epsilon-greedy (GLIE) with extra mass on under-explored
  entities; epsilon decays linearly to a small floor.

Zero data leaks across seeds: all state lives exclusively in instance variables
created in __init__.
"""

import heapq
import random
import re
import time

import ext_elev

id = ["331050591"]

INF = float("inf")

# ── Bayesian prior for success probs: Beta(alpha, beta) → mean = alpha/(a+b)
_PA = 9.0   # prior "successes"   → prior mean 0.9
_PB = 1.0   # prior "failures"
_MIN_TRIES   = 8       # samples below this → entity treated as under-explored
_REBUILD_DIV = 20      # rebuild every max_steps//_REBUILD_DIV steps (min 15)

_ACTION_RE = re.compile(
    r'\s*(MOVE|ENTER|EXIT)\s*\{\s*(-?\d+)\s*,\s*(-?\d+)\s*\}\s*'
)


# ══════════════════════════════════════════════════════════════════════════════
# Plan  (verbatim from ex2)
# ══════════════════════════════════════════════════════════════════════════════

class Plan:
    """Delivery path for a person: ordered (elevator, pickup_floor, dropoff_floor) legs."""
    __slots__ = ("legs", "intr", "suffix", "intr_min", "suffix_min")

    def __init__(self, legs, pe, q):
        self.legs = legs
        n = len(legs)
        self.intr = []
        for (e, a, b) in legs:
            c = 2.0 / q
            if a != b:
                c += 1.0 / pe[e]
            self.intr.append(c)
        self.suffix = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            self.suffix[i] = self.suffix[i + 1] + 1.0 / pe[legs[i][0]] + self.intr[i]
        self.intr_min = [2 + (1 if a != b else 0) for (e, a, b) in legs]
        self.suffix_min = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            self.suffix_min[i] = self.suffix_min[i + 1] + 1 + self.intr_min[i]


# ══════════════════════════════════════════════════════════════════════════════
# _JointAstar  (verbatim from ex2)
# ══════════════════════════════════════════════════════════════════════════════

class _JointAstar:
    _DEL = -1

    def __init__(self, ctrl, plan_eids=None):
        c = ctrl
        eids = plan_eids if plan_eids is not None else tuple(c.elev_ids)
        self._eids   = eids
        self._pids   = c.person_ids
        self._eidx   = {e: i for i, e in enumerate(eids)}
        self._pidx   = {p: j for j, p in enumerate(c.person_ids)}
        self._goal   = tuple(c.goal_floor[p] for p in c.person_ids)
        self._wt     = tuple(c.weight[p]     for p in c.person_ids)
        self._cap    = tuple(c.cap[e]        for e in eids)
        self._reach  = tuple(c.reachable[e]  for e in eids)
        self._qp     = tuple(c.qp[p]         for p in c.person_ids)
        pe_raw       = tuple(c.pe[e]          for e in eids)
        self._pe_cost = tuple(1.0 / (p ** 1.3) for p in pe_raw)
        all_floors = set(c.init_efloor.values())
        for r in self._reach:
            all_floors |= r
        self._OFF = max(all_floors) + 1 if all_floors else 1
        self._shared = frozenset(
            f for f in all_floors
            if sum(1 for r in self._reach if f in r) >= 2
        )
        self._cache: dict = {}
        self._max_expand = 60000

    def get_action(self, efl, ploc, target):
        ef  = tuple(efl[e] for e in self._eids)
        pl  = self._encode(ploc, target)
        if pl is None or all(x == self._DEL for x in pl):
            return None
        key = (ef, pl)
        act = self._cache.get(key)
        if act is not None:
            return act
        path = self._search(ef, pl)
        if not path:
            return None
        for st, a in path:
            self._cache.setdefault(st, a)
        return path[0][1]

    def _encode(self, ploc, target):
        OFF, DEL = self._OFF, self._DEL
        pl = [DEL] * len(self._pids)
        for p in target:
            loc = ploc.get(p)
            if loc is None:
                continue
            j = self._pidx[p]
            if loc[0] == 'floor':
                pl[j] = loc[1]
            else:
                e = loc[1]
                if e not in self._eidx:
                    return None
                pl[j] = OFF + self._eidx[e]
        return tuple(pl)

    def _h(self, ef, pl):
        OFF, DEL = self._OFF, self._DEL
        total = 0.0
        elev_move = [False] * len(self._eids)
        for j, loc in enumerate(pl):
            if loc == DEL:
                continue
            if loc >= OFF:
                i = loc - OFF
                total += 1.0 / self._qp[j]
                if ef[i] != self._goal[j]:
                    elev_move[i] = True
            else:
                total += 2.0 / self._qp[j]
                f = loc
                for i, r in enumerate(self._reach):
                    if f in r and self._wt[j] <= self._cap[i]:
                        if ef[i] != f:
                            elev_move[i] = True
                        break
        for i, need in enumerate(elev_move):
            if need:
                total += self._pe_cost[i]
        return total

    def _successors(self, ef, pl):
        OFF, DEL = self._OFF, self._DEL
        NE = len(self._eids)
        loads = [0] * NE
        for j, loc in enumerate(pl):
            if loc != DEL and loc >= OFF:
                loads[loc - OFF] += self._wt[j]
        for j, loc in enumerate(pl):
            if loc == DEL or loc < OFF:
                continue
            i = loc - OFF
            if ef[i] == self._goal[j]:
                npl = pl[:j] + (self._DEL,) + pl[j + 1:]
                yield (f'EXIT{{{self._pids[j]},{self._eids[i]}}}',
                       ef, npl, 1.0 / self._qp[j])
                return
        for j, loc in enumerate(pl):
            if loc == DEL or loc < OFF:
                continue
            i = loc - OFF
            if self._goal[j] in self._reach[i]:
                continue
            f = ef[i]
            if f not in self._shared:
                continue
            if any(k != i and f in self._reach[k] and self._wt[j] <= self._cap[k]
                   for k in range(NE)):
                npl = pl[:j] + (f,) + pl[j + 1:]
                yield (f'EXIT{{{self._pids[j]},{self._eids[i]}}}',
                       ef, npl, 1.0 / self._qp[j])
        for j, loc in enumerate(pl):
            if loc == DEL or loc >= OFF:
                continue
            f = loc
            for i in range(NE):
                if (ef[i] == f and f in self._reach[i]
                        and self._wt[j] + loads[i] <= self._cap[i]):
                    npl = pl[:j] + (OFF + i,) + pl[j + 1:]
                    yield (f'ENTER{{{self._pids[j]},{self._eids[i]}}}',
                           ef, npl, 1.0 / self._qp[j])
        move_targets: dict = {}
        for j, loc in enumerate(pl):
            if loc == DEL:
                continue
            if loc >= OFF:
                i = loc - OFF
                g = self._goal[j]
                if g in self._reach[i] and g != ef[i]:
                    move_targets.setdefault(i, set()).add(g)
                elif g not in self._reach[i]:
                    for f in self._shared:
                        if f in self._reach[i] and f != ef[i]:
                            move_targets.setdefault(i, set()).add(f)
            else:
                f = loc
                for i, r in enumerate(self._reach):
                    if f in r and self._wt[j] <= self._cap[i] and ef[i] != f:
                        move_targets.setdefault(i, set()).add(f)
        for i, floors in move_targets.items():
            for f in floors:
                nef = ef[:i] + (f,) + ef[i + 1:]
                yield (f'MOVE{{{self._eids[i]},{f}}}',
                       nef, pl, self._pe_cost[i])

    def _search(self, ef0, pl0):
        DEL   = self._DEL
        start = (ef0, pl0)
        costs = {start: 0.0}
        parent: dict = {start: None}
        cnt, expanded = 0, 0
        heap = [(self._h(ef0, pl0), 0.0, cnt, start)]
        while heap:
            _, g, _, state = heapq.heappop(heap)
            if g > costs.get(state, INF) + 1e-9:
                continue
            ef, pl = state
            if all(x == DEL for x in pl):
                path = []
                cur  = state
                while parent[cur] is not None:
                    prev, act = parent[cur]
                    path.append((prev, act))
                    cur = prev
                path.reverse()
                return path
            expanded += 1
            if expanded > self._max_expand:
                return None
            for act, nef, npl, cost in self._successors(ef, pl):
                ns = (nef, npl)
                ng = g + cost
                if ng >= costs.get(ns, INF) - 1e-9:
                    continue
                costs[ns] = ng
                parent[ns] = (state, act)
                cnt += 1
                heapq.heappush(heap, (ng + self._h(nef, npl), ng, cnt, ns))
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Controller
# ══════════════════════════════════════════════════════════════════════════════

class Controller:
    """ADP + GLIE multi-elevator controller.

    Learns pe (elevator), qp (person), and Erew (delivery reward) empirically
    using Bayesian counters.  Feeds the estimates to the Assignment-2 planner
    (expectimax / A*).  GLIE exploration via decaying epsilon-greedy with bias
    toward under-explored entities.
    """

    def __init__(self, game: ext_elev.GameAPI):
        self.game = game

        init_state = game.get_initial_state()
        elevators_t, persons_t, _ = init_state
        self.initial_state = init_state
        self.max_steps  = game.get_max_steps()
        self.goal_reward = float(game.get_goal_reward())

        reachable  = game.get_reachable()
        capacities = game.get_capacities()

        self.elev_ids  = [eid for (eid, _, _) in elevators_t]
        self.reachable = {e: frozenset(reachable[e]) for e in self.elev_ids}
        self.cap       = {e: capacities[e]           for e in self.elev_ids}
        self.init_efloor = {eid: fl for (eid, fl, _) in elevators_t}

        self.person_ids  = [pid for (pid, _) in persons_t]
        self.start       = {pid: loc[1] for (pid, loc) in persons_t}
        self.goal_floor  = {p: game.get_person_goal(p)  for p in self.person_ids}
        self.weight      = {p: game.get_person_weight(p) for p in self.person_ids}
        self.all_persons = frozenset(self.person_ids)

        # ── ADP counters ──────────────────────────────────────────────────
        self._etries = {e: 0 for e in self.elev_ids}
        self._esucc  = {e: 0 for e in self.elev_ids}
        self._ptries = {p: 0 for p in self.person_ids}
        self._psucc  = {p: 0 for p in self.person_ids}
        self._rsum   = {p: 0.0 for p in self.person_ids}
        self._rcnt   = {p: 0   for p in self.person_ids}

        # ── Initial empirical estimates (Beta prior mean = 0.9) ───────────
        self.pe   = {e: _PA / (_PA + _PB) for e in self.elev_ids}
        self.qp   = {p: _PA / (_PA + _PB) for p in self.person_ids}
        # Reward prior: aim slightly below goal_reward/n (conservative)
        n_p = max(1, len(self.person_ids))
        self._rew_prior = max(4.0, self.goal_reward / n_p * 0.8)
        self.Erew = {p: self._rew_prior for p in self.person_ids}

        # ── Last-action bookkeeping ────────────────────────────────────────
        self._last_action = None   # str
        self._last_state  = None   # state tuple

        # ── Planner state (rebuilt periodically) ──────────────────────────
        self.plans          = {}
        self.deliverable    = set()
        self.target         = frozenset()
        self.allpersons_flag = False
        self.rho             = 0.0
        self._plan_eids      = tuple(self.elev_ids)
        self._cache          = {}
        self._use_astar      = False
        self._astar_planner  = None
        self._max_depth      = 4
        self.min_steps_init  = {}

        self._rebuild_interval = max(15, self.max_steps // _REBUILD_DIV)
        self._last_rebuild     = -self._rebuild_interval  # force immediate
        self._needs_rebuild    = True

        self._t0 = time.perf_counter()
        # Grader allows 20 + 0.5*horizon per seed; use half to leave margin
        # for rebuild overhead (not time-regulated) on slow university servers.
        self._time_budget = 10.0 + 0.25 * self.max_steps

        self._rebuild_planner_state()

    # ── ADP: model update ─────────────────────────────────────────────────────

    def _update_model(self, curr_state):
        """Update empirical estimates from the outcome of the previous action."""
        if self._last_action is None or self._last_state is None:
            return
        if self._last_action == "RESET":
            return

        m = _ACTION_RE.fullmatch(self._last_action)
        if not m:
            return
        kind, a, b = m.group(1), int(m.group(2)), int(m.group(3))

        last_elev_t, last_pers_t, last_remaining = self._last_state
        curr_elev_t, curr_pers_t, _ = curr_state
        last_efl  = {eid: fl for (eid, fl, _) in last_elev_t}
        curr_efl  = {eid: fl for (eid, fl, _) in curr_elev_t}
        curr_ploc = dict(curr_pers_t)

        if kind == 'MOVE':
            e, target_f = a, b
            self._etries[e] += 1
            if curr_efl.get(e) == target_f:
                self._esucc[e] += 1
            n, s = self._etries[e], self._esucc[e]
            self.pe[e] = (s + _PA) / (n + _PA + _PB)
            if n in (5, 10, 20, 50):
                self._needs_rebuild = True

        elif kind == 'ENTER':
            p, e = a, b
            self._ptries[p] += 1
            if curr_ploc.get(p) == ('in', e):
                self._psucc[p] += 1
            n, s = self._ptries[p], self._psucc[p]
            self.qp[p] = (s + _PA) / (n + _PA + _PB)
            if n in (5, 10, 20, 50):
                self._needs_rebuild = True

        else:  # EXIT
            p, e = a, b
            self._ptries[p] += 1
            last_f     = last_efl.get(e)
            delivered  = (p not in curr_ploc)
            exited_mid = (not delivered and curr_ploc.get(p) == ('floor', last_f))

            if delivered or exited_mid:
                self._psucc[p] += 1

            if delivered:
                raw = float(self.game.get_last_gained_reward())
                if last_remaining == 1:
                    raw = max(0.0, raw - self.goal_reward)
                else:
                    raw = max(0.0, raw)
                self._rsum[p] += raw
                self._rcnt[p] += 1
                self.Erew[p] = self._rsum[p] / self._rcnt[p]
                self._needs_rebuild = True

            n, s = self._ptries[p], self._psucc[p]
            self.qp[p] = (s + _PA) / (n + _PA + _PB)
            if n in (5, 10, 20, 50):
                self._needs_rebuild = True

    # ── Planner rebuild ───────────────────────────────────────────────────────

    def _rebuild_planner_state(self):
        """Re-derive plans/cycle/A* from current pe/qp/Erew estimates."""
        self.plans       = {}
        self.deliverable = set()
        for p in self.person_ids:
            plist = self._build_plans(p)
            self.plans[p] = plist
            if plist:
                self.deliverable.add(p)

        self.min_steps_init = {
            p: (self._person_min_steps(p, ('floor', self.start[p]), self.init_efloor)
                if self.plans[p] else INF)
            for p in self.person_ids
        }

        self.target, self.allpersons_flag, self.rho = self._choose_cycle()

        anchor_cands = [
            e for e in self.elev_ids
            if all(
                self.start[p] in self.reachable[e]
                and self.goal_floor[p] in self.reachable[e]
                and self.weight[p] <= self.cap[e]
                for p in self.target
            )
        ]
        self._plan_eids = (
            (max(anchor_cands, key=lambda e: self.pe[e]),)
            if anchor_cands else tuple(self.elev_ids)
        )

        n_t = len(self.target)
        any_relay = any(
            min((len(pl.legs) for pl in self.plans[p]), default=1) > 1
            for p in self.target
        )
        if n_t >= 5 and any_relay:
            self._max_depth = 3
        elif n_t >= 4:
            self._max_depth = 4
        elif any_relay:
            self._max_depth = 6  # relay but few persons: moderate lookahead
        else:
            self._max_depth = 4  # direct-delivery: depth 4 is sufficient

        relay_persons = [p for p in self.target if
                         min((len(pl.legs) for pl in self.plans[p]), default=1) > 1]
        all_batchable = all(
            any(self.weight[pa] + self.weight[pb] <= self.cap[e]
                for e in self.elev_ids)
            for i, pa in enumerate(relay_persons)
            for pb in relay_persons[i + 1:]
        ) if len(relay_persons) >= 2 else True
        any_broken = any(self.pe[e] < 0.5 for e in self.elev_ids)
        self._use_astar = (
            self.allpersons_flag and n_t >= 4
            and any_relay
            and len(self._plan_eids) <= 2
            and (all_batchable or not any_broken)
        )
        if self._use_astar:
            self._astar_planner = _JointAstar(self, plan_eids=self._plan_eids)

        self._cache = {}

    # ── Plan construction (from ex2) ──────────────────────────────────────────

    def _build_plans(self, p):
        s, g, w = self.start[p], self.goal_floor[p], self.weight[p]
        seen  = set()
        plans = []
        for e in self.elev_ids:
            s_ok = s in self.reachable[e] or s == self.init_efloor[e]
            if s_ok and g in self.reachable[e] and w <= self.cap[e]:
                legs = ((e, s, g),)
                if legs not in seen:
                    seen.add(legs)
                    plans.append(Plan(list(legs), self.pe, self.qp[p]))
        relay = self._dijkstra_plan(p)
        if relay is not None:
            key = tuple(relay)
            if key not in seen:
                seen.add(key)
                plans.append(Plan(relay, self.pe, self.qp[p]))
        return plans

    def _dijkstra_plan(self, p):
        s, g, w, q = self.start[p], self.goal_floor[p], self.weight[p], self.qp[p]
        start_node = ('F', s)
        dist  = {start_node: 0.0}
        prev  = {start_node: None}
        cnt   = 0
        heap  = [(0.0, cnt, start_node)]
        while heap:
            d, _, node = heapq.heappop(heap)
            if d > dist.get(node, INF) + 1e-9:
                continue
            if node == 'DONE':
                break
            for nbr, cost in self._dijkstra_edges(node, g, w, q):
                nd = d + cost
                if nd < dist.get(nbr, INF) - 1e-9:
                    dist[nbr] = nd
                    prev[nbr] = node
                    cnt += 1
                    heapq.heappush(heap, (nd, cnt, nbr))
        if 'DONE' not in prev:
            return None
        path = []
        cur  = 'DONE'
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        legs = []
        cur_e = cur_a = cur_f = None
        for node in path:
            if node == 'DONE':
                if cur_e is not None:
                    legs.append((cur_e, cur_a, cur_f))
                break
            if node[0] == 'F':
                if cur_e is not None:
                    legs.append((cur_e, cur_a, cur_f))
                    cur_e = None
                cur_f = node[1]
            else:
                _, e, f = node
                if cur_e is None:
                    cur_e, cur_a = e, f
                cur_f = f
        return legs if legs else None

    def _dijkstra_edges(self, node, goal, weight, q):
        out = []
        if node[0] == 'F':
            f = node[1]
            for e in self.elev_ids:
                if (f in self.reachable[e] or f == self.init_efloor[e]) and weight <= self.cap[e]:
                    out.append((('I', e, f), 1.0 / self.pe[e] + 1.0 / q))
        else:
            _, e, f = node
            for f2 in self.reachable[e]:
                if f2 != f:
                    out.append((('I', e, f2), 1.0 / self.pe[e]))
            if f == goal:
                out.append(('DONE', 1.0 / q))
            else:
                out.append((('F', f), 1.0 / q))
        return out

    # ── Cycle selection (from ex2) ────────────────────────────────────────────

    def _choose_cycle(self):
        deliverable = sorted(self.deliverable)
        n = len(deliverable)
        if n == 0:
            return frozenset(), False, 0.0
        best_rate, best_target, best_allflag = -INF, frozenset(), False
        for mask in range(1, 1 << n):
            S       = frozenset(deliverable[i] for i in range(n) if (mask >> i) & 1)
            allflag = (S == self.all_persons)
            reward  = sum(self.Erew[p] for p in S)
            if allflag:
                reward += self.goal_reward
            cost = self._estimate_cycle_cost(S, allflag)
            if cost <= 0 or cost == INF:
                continue
            rate = reward / cost
            if rate > best_rate:
                best_rate, best_target, best_allflag = rate, S, allflag
        return best_target, best_allflag, max(best_rate, 0.0)

    def _estimate_cycle_cost(self, S_set, allflag):
        efl  = dict(self.init_efloor)
        ew   = {e: 0 for e in self.elev_ids}
        ploc = {p: ('floor', self.start[p]) for p in self.person_ids}
        cost = 0.0
        for _ in range(600):
            present = [p for p in S_set if p in ploc]
            if not present:
                if not allflag:
                    cost += 1.0
                return cost
            best_act = best_prob = None
            best_after = INF
            for act, prob in self._candidate_actions(efl, ew, ploc, present):
                if act[0] == 'RESET':
                    continue
                nefl, nw, nploc = self._apply_success(act, efl, ew, ploc)
                ng = sum(self._ctg(p, nploc[p], nefl) for p in S_set if p in nploc)
                if ng < best_after - 1e-9:
                    best_after, best_act, best_prob = ng, act, prob
            if best_act is None:
                return INF
            cost += 1.0 / best_prob
            efl, ew, ploc = self._apply_success(best_act, efl, ew, ploc)
        return INF

    # ── Cost-to-go helpers (from ex2) ─────────────────────────────────────────

    def _ctg(self, p, loc, efl):
        best = INF
        if loc[0] == 'in':
            e, f = loc[1], efl[loc[1]]
            for plan in self.plans[p]:
                for i, (le, a, b) in enumerate(plan.legs):
                    if le != e:
                        continue
                    c = (0.0 if f == b else 1.0 / self.pe[e]) + 1.0 / self.qp[p] + plan.suffix[i + 1]
                    if c < best:
                        best = c
                    break
        else:
            f = loc[1]
            for plan in self.plans[p]:
                for i, (le, a, b) in enumerate(plan.legs):
                    if a != f:
                        continue
                    if efl[le] == f:
                        repo = 0.0
                    elif f in self.reachable[le]:
                        repo = 1.0 / self.pe[le]
                    else:
                        break
                    c = repo + plan.intr[i] + plan.suffix[i + 1]
                    if c < best:
                        best = c
                    break
        return best

    def _person_min_steps(self, p, loc, efl):
        best = INF
        if loc[0] == 'in':
            e, f = loc[1], efl[loc[1]]
            for plan in self.plans[p]:
                for i, (le, a, b) in enumerate(plan.legs):
                    if le != e:
                        continue
                    c = (0 if f == b else 1) + 1 + plan.suffix_min[i + 1]
                    if c < best:
                        best = c
                    break
        else:
            f = loc[1]
            for plan in self.plans[p]:
                for i, (le, a, b) in enumerate(plan.legs):
                    if a != f:
                        continue
                    if efl[le] == f:
                        repo = 0
                    elif f in self.reachable[le]:
                        repo = 1
                    else:
                        break
                    c = repo + plan.intr_min[i] + plan.suffix_min[i + 1]
                    if c < best:
                        best = c
                    break
        return best

    def _potential(self, efl, ploc, steps_left):
        R = g = 0.0
        n_present = n_pursuable = 0
        reposition_group = {}
        for p in self.target:
            loc = ploc.get(p)
            if loc is None:
                continue
            n_present += 1
            if self._person_min_steps(p, loc, efl) <= steps_left:
                g += self._ctg(p, loc, efl)
                R += self.Erew[p]
                n_pursuable += 1
                if loc[0] == 'floor':
                    f = loc[1]
                    for plan in self.plans[p]:
                        for (le, a, b) in plan.legs:
                            if a != f:
                                continue
                            if efl[le] != f and f in self.reachable[le]:
                                key = (le, f)
                                reposition_group.setdefault(key, []).append(self.weight[p])
                            break
        redundancy_save = 0.0
        for (le, f), weights in reposition_group.items():
            if len(weights) < 2:
                continue
            ws = sorted(weights)
            if ws[0] + ws[1] <= self.cap[le]:
                redundancy_save += (len(weights) - 1) / self.pe[le]
        if self.allpersons_flag and n_present > 0 and n_pursuable == n_present:
            R += self.goal_reward
        return R - self.rho * (g - redundancy_save)

    # ── Action generation and transition model (from ex2) ─────────────────────

    def _candidate_actions(self, efl, ew, ploc, present):
        acts = []
        seen = set()

        def add(a, prob):
            if a not in seen:
                seen.add(a)
                acts.append((a, prob))

        for p in present:
            loc = ploc[p]
            if loc[0] == 'in':
                e, f = loc[1], efl[loc[1]]
                for plan in self.plans[p]:
                    for (le, a, b) in plan.legs:
                        if le != e:
                            continue
                        if f == b:
                            add(('EXIT', p, e), self.qp[p])
                        else:
                            add(('MOVE', e, b), self.pe[e])
                        break
            else:
                f = loc[1]
                for plan in self.plans[p]:
                    for (le, a, b) in plan.legs:
                        if a != f:
                            continue
                        if efl[le] == f:
                            if ew[le] + self.weight[p] <= self.cap[le]:
                                add(('ENTER', p, le), self.qp[p])
                        elif f in self.reachable[le]:
                            add(('MOVE', le, f), self.pe[le])
                        break

        acts.append((('RESET',), 1.0))
        return acts

    def _apply_success(self, act, efl, ew, ploc):
        kind = act[0]
        if kind == 'RESET':
            return (dict(self.init_efloor),
                    {e: 0 for e in self.elev_ids},
                    {p: ('floor', self.start[p]) for p in self.person_ids})
        nefl, nw, nploc = dict(efl), dict(ew), dict(ploc)
        if kind == 'MOVE':
            nefl[act[1]] = act[2]
        elif kind == 'ENTER':
            nploc[act[1]] = ('in', act[2])
            nw[act[2]] += self.weight[act[1]]
        else:
            p, e = act[1], act[2]
            f = nefl[e]
            nw[e] -= self.weight[p]
            if f == self.goal_floor[p]:
                del nploc[p]
            else:
                nploc[p] = ('floor', f)
        return nefl, nw, nploc

    def _outcomes(self, act, efl, ew, ploc):
        kind = act[0]
        if kind == 'RESET':
            yield (1.0, 0.0,
                   dict(self.init_efloor),
                   {e: 0 for e in self.elev_ids},
                   {p: ('floor', self.start[p]) for p in self.person_ids})
            return
        if kind == 'MOVE':
            _, e, target = act
            pe  = self.pe[e]
            nef = dict(efl); nef[e] = target
            yield (pe, 0.0, nef, ew, ploc)
            cur = efl[e]
            fail_opts = sorted({cur} | (set(self.reachable[e]) - {target}))
            pf = (1.0 - pe) / len(fail_opts)
            for fo in fail_opts:
                nf = dict(efl); nf[e] = fo
                yield (pf, 0.0, nf, ew, ploc)
            return
        if kind == 'ENTER':
            _, p, e = act
            q = self.qp[p]
            npl = dict(ploc); npl[p] = ('in', e)
            nw  = dict(ew);   nw[e] += self.weight[p]
            yield (q, 0.0, efl, nw, npl)
            yield (1.0 - q, 0.0, efl, ew, ploc)
            return
        # EXIT
        _, p, e = act
        q  = self.qp[p]
        f  = efl[e]
        nw = dict(ew); nw[e] -= self.weight[p]
        if f == self.goal_floor[p]:
            npl = dict(ploc); del npl[p]
            r   = self.Erew[p]
            if len(npl) == 0:
                yield (q, r + self.goal_reward,
                       dict(self.init_efloor),
                       {ee: 0 for ee in self.elev_ids},
                       {pp: ('floor', self.start[pp]) for pp in self.person_ids})
            else:
                yield (q, r, efl, nw, npl)
        else:
            npl = dict(ploc); npl[p] = ('floor', f)
            yield (q, 0.0, efl, nw, npl)
        yield (1.0 - q, 0.0, efl, ew, ploc)

    # ── Expectimax (from ex2) ─────────────────────────────────────────────────

    def _canon(self, efl, ploc):
        return (tuple(sorted(efl.items())), tuple(sorted(ploc.items())))

    def _value(self, efl, ew, ploc, depth, steps_left):
        if depth <= 0 or steps_left <= 0:
            return self._potential(efl, ploc, steps_left)
        key = (self._canon(efl, ploc), depth)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        present = [p for p in self.target if p in ploc]
        acts    = self._candidate_actions(efl, ew, ploc, present)
        best    = -INF
        for act, _ in acts:
            ev = 0.0
            for prob, r, nefl, nw, nploc in self._outcomes(act, efl, ew, ploc):
                ev += prob * (r + self._value(nefl, nw, nploc, depth - 1, steps_left - 1))
            if ev > best:
                best = ev
        if best == -INF:
            best = self._potential(efl, ploc, steps_left)
        self._cache[key] = best
        return best

    def _endgame_eval(self, efl, ew, ploc, all_persons, steps_left):
        saved = self.target, self.allpersons_flag, self.rho
        self.target         = frozenset(all_persons)
        self.allpersons_flag = (self.target == self.all_persons)
        self.rho             = 0.0
        self._cache          = {}
        depth = min(steps_left, 15)
        acts  = self._candidate_actions(efl, ew, ploc, list(all_persons))
        PRIORITY = {'EXIT': 3, 'ENTER': 2, 'MOVE': 1, 'RESET': 0}
        best_act = best_key = None
        best_val = -INF
        for act, prob in acts:
            ev = g_ev = 0.0
            for pr, r, nefl, nw, nploc in self._outcomes(act, efl, ew, ploc):
                ev += pr * (r + self._value(nefl, nw, nploc, depth - 1, steps_left - 1))
                for p in self.target:
                    loc = nploc.get(p)
                    if loc is not None and self._person_min_steps(p, loc, nefl) <= steps_left - 1:
                        g_ev += pr * self._ctg(p, loc, nefl)
            key = (-g_ev, PRIORITY[act[0]], prob)
            if best_act is None or ev > best_val + 1e-9:
                best_act, best_val, best_key = act, ev, key
            elif ev >= best_val - 1e-9 and key > best_key:
                best_act, best_key = act, key
        self.target, self.allpersons_flag, self.rho = saved
        return best_act

    # ── Tail-subset helper (from ex2) ─────────────────────────────────────────

    def _tail_subset(self, efl, ploc, present, steps_left):
        plan_set = set(self._plan_eids)
        for p in present:
            if ploc[p][0] == 'in' and ploc[p][1] not in plan_set:
                return None
        min_each = {p: self._person_min_steps(p, ploc[p], efl) for p in present}
        if sum(min_each.values()) <= steps_left:
            return None
        boarded = [p for p in present if ploc[p][0] == 'in']
        if boarded and sum(min_each[p] for p in boarded) > steps_left:
            return None
        waiting = [p for p in present if ploc[p][0] != 'in']
        n = len(waiting)
        if n == 0 or n > 10:
            return None
        candidates = []
        for mask in range(1 << n):
            subset = boarded + [waiting[i] for i in range(n) if mask >> i & 1]
            if not subset or len(subset) == len(present):
                continue
            ms  = sum(min_each[p] for p in subset)
            rew = sum(self.Erew[p] for p in subset)
            candidates.append((rew, -ms, subset))
        candidates.sort(reverse=True)
        for _, _, subset in candidates:
            if sum(min_each[p] for p in subset) <= steps_left:
                return frozenset(subset)
        return None

    # ── Fallback / format (from ex2) ─────────────────────────────────────────

    def _fallback(self, elevators_t):
        for (eid, cur_f, _) in elevators_t:
            for f in self.reachable[eid]:
                if f != cur_f:
                    return f"MOVE{{{eid},{f}}}"
        return "RESET"

    def _format(self, act):
        k = act[0]
        if k == 'RESET':  return "RESET"
        if k == 'MOVE':   return f"MOVE{{{act[1]},{act[2]}}}"
        if k == 'ENTER':  return f"ENTER{{{act[1]},{act[2]}}}"
        return f"EXIT{{{act[1]},{act[2]}}}"

    # ── Main entry point ──────────────────────────────────────────────────────

    def choose_next_action(self, state):
        # ── 1. Update empirical model from last action's outcome ──────────
        self._update_model(state)

        t          = self.game.get_current_steps()
        steps_left = self.max_steps - t

        # ── 2. Periodic planner rebuild ───────────────────────────────────
        if self._needs_rebuild or (t - self._last_rebuild) >= self._rebuild_interval:
            self._rebuild_planner_state()
            self._last_rebuild  = t
            self._needs_rebuild = False

        elevators_t, persons_t, _ = state
        efl  = {eid: fl  for (eid, fl, _) in elevators_t}
        ew   = {eid: w   for (eid, _, w)  in elevators_t}
        ploc = {pid: loc for (pid, loc)   in persons_t}

        present     = [p for p in self.target      if p in ploc]
        all_present = [p for p in self.person_ids  if p in ploc]

        # ── 3. GLIE exploration ───────────────────────────────────────────
        # Epsilon decays linearly from 0.30 to 0.04 over the horizon.
        epsilon = max(0.04, 0.30 * (1.0 - t / self.max_steps))

        if random.random() < epsilon:
            # Collect legal candidates covering all present persons
            cands_for_glie = self._candidate_actions(
                efl, ew, ploc, present if present else all_present
            )
            # Prefer actions on under-explored entities
            under = [
                self._format(act) for act, _ in cands_for_glie
                if (act[0] == 'MOVE'  and self._etries.get(act[1], 0) < _MIN_TRIES)
                or (act[0] in ('ENTER', 'EXIT')
                    and self._ptries.get(act[1], 0) < _MIN_TRIES)
            ]
            pool = under if under else [
                self._format(act) for act, _ in cands_for_glie
                if act[0] != 'RESET'
            ]
            if pool:
                action = random.choice(pool)
                self._last_state  = state
                self._last_action = action
                return action

        # ── 4. Exploit: planner (identical to ex2 choose_next_action) ─────

        # Fast path: A* joint planner
        if self._use_astar:
            if present:
                sub = self._tail_subset(efl, ploc, present, steps_left)
                if sub is not None:
                    act = self._astar_planner.get_action(efl, ploc, sub)
                    if act is not None:
                        self._last_state  = state
                        self._last_action = act
                        return act
            act = self._astar_planner.get_action(efl, ploc, self.target)
            if act is not None:
                self._last_state  = state
                self._last_action = act
                return act

        # Cycle complete: decide whether to RESET
        if not present:
            in_transit = [p for p in self.person_ids
                          if p not in self.target and p in ploc
                          and ploc[p][0] == 'in']
            if in_transit:
                act = self._endgame_eval(efl, ew, ploc,
                                         [p for p in self.person_ids if p in ploc],
                                         steps_left)
                if act is not None:
                    action = self._format(act)
                    self._last_state  = state
                    self._last_action = action
                    return action
            if self.target:
                cheapest = min((self.min_steps_init[p] for p in self.target), default=INF)
                if steps_left >= 1 + cheapest:
                    self._last_state  = state
                    self._last_action = "RESET"
                    return "RESET"
            action = self._fallback(elevators_t)
            self._last_state  = state
            self._last_action = action
            return action

        # Endgame: widen to all persons
        min_cycle = min(
            (self._person_min_steps(p, ploc[p], efl) for p in present), default=INF
        )
        if self.target != self.all_persons:
            min_cycle += 1
        if min_cycle < INF and steps_left < 2 * min_cycle:
            all_rem = [p for p in self.person_ids if p in ploc]
            if len(all_rem) > len(present):
                act = self._endgame_eval(efl, ew, ploc, all_rem, steps_left)
                if act is not None:
                    action = self._format(act)
                    self._last_state  = state
                    self._last_action = action
                    return action

        # Iterative-deepening expectimax
        elapsed     = time.perf_counter() - self._t0
        remaining   = max(0.0, self._time_budget - elapsed)
        step_budget = min(0.5, remaining / max(steps_left, 1))

        acts_exploit = self._candidate_actions(efl, ew, ploc, present)
        PRIORITY     = {'EXIT': 3, 'ENTER': 2, 'MOVE': 1, 'RESET': 0}
        best_act     = None
        t_step       = time.perf_counter()

        for depth in range(1, self._max_depth + 1):
            self._cache = {}
            t_depth     = time.perf_counter()
            curr_act    = None
            curr_val    = -INF
            curr_key    = None

            for act, prob in acts_exploit:
                ev = g_ev = 0.0
                for pr, rv, nefl, nw, nploc in self._outcomes(act, efl, ew, ploc):
                    ev += pr * (rv + self._value(nefl, nw, nploc, depth - 1, steps_left - 1))
                    for p in self.target:
                        loc = nploc.get(p)
                        if loc is not None and self._person_min_steps(p, loc, nefl) <= steps_left - 1:
                            g_ev += pr * self._ctg(p, loc, nefl)
                key = (-g_ev, PRIORITY[act[0]], prob)
                if curr_act is None or ev > curr_val + 1e-9:
                    curr_act, curr_val, curr_key = act, ev, key
                elif ev >= curr_val - 1e-9 and key > curr_key:
                    curr_act, curr_key = act, key

            best_act   = curr_act
            depth_time = time.perf_counter() - t_depth
            step_time  = time.perf_counter() - t_step
            if step_time + depth_time * 4 > step_budget:
                break

        action = self._format(best_act) if best_act else self._fallback(elevators_t)
        self._last_state  = state
        self._last_action = action
        return action
