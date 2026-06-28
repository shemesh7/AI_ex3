"""Quick benchmark: 5 seeds, representative subset of problems."""
import sys, time
import ext_elev
import ex3_331050591 as ex3

def solve(problem, seed):
    problem = dict(problem, seed=seed)
    api = ext_elev.create_elevators_game(problem, debug=False)
    ctrl = ex3.Controller(api)
    for _ in range(api.get_max_steps()):
        action = ctrl.choose_next_action(api.get_current_state())
        api.submit_next_action(action)
        if api.get_done():
            break
    return api.get_current_reward()

from ex3_check import (
    problem_p1_easy, problem_e3_easy, problem_m1_easy, problem_rl_easy,
    problem_p1_med,  problem_e3_med,  problem_m1_med,  problem_rl_med,
    problem_p1_hard, problem_e3_hard, problem_m1_hard, problem_rl_hard,
)

PROBLEMS = [
    ("p1_easy",  problem_p1_easy),
    ("e3_easy",  problem_e3_easy),
    ("m1_easy",  problem_m1_easy),
    ("rl_easy",  problem_rl_easy),
    ("p1_med",   problem_p1_med),
    ("e3_med",   problem_e3_med),
    ("m1_med",   problem_m1_med),
    ("rl_med",   problem_rl_med),
    ("p1_hard",  problem_p1_hard),
    ("e3_hard",  problem_e3_hard),
    ("m1_hard",  problem_m1_hard),
    ("rl_hard",  problem_rl_hard),
]

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
total = 0.0
for name, prob in PROBLEMS:
    rewards = [solve(prob, s) for s in range(N)]
    avg = sum(rewards) / N
    total += avg
    print(f"{name:12s}  avg={avg:7.1f}  {rewards}")
print(f"\nTOTAL avg sum = {total:.1f}")
