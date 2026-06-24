import re
from pathlib import Path

BASELINES = ["random", "explore_vi", "explore_astar", "td", "thompson"]

PROBLEM_ORDER = [
    "p1_easy", "e1_easy", "e2_easy", "e3_easy", "e4_easy", "e5_easy",
    "m1_easy", "m2_easy", "m3_easy", "m4_easy", "m5_easy", "rl_easy",
    "p1_med",  "e1_med",  "e2_med",  "e3_med",  "e4_med",  "e5_med",
    "m1_med",  "m2_med",  "m3_med",  "m4_med",  "m5_med",  "rl_med",
    "p1_hard", "e1_hard", "e2_hard", "e3_hard", "e4_hard", "e5_hard",
    "m1_hard", "m2_hard", "m3_hard", "m4_hard", "m5_hard", "rl_hard",
]

base_dir = Path(__file__).parent


def parse_solution(path):
    """Parse Solution.txt, keeping the last value for each problem."""
    results = {}
    pattern = re.compile(r"^(\w+):\s*reward_average=([\d.]+)")
    for line in Path(path).read_text().splitlines():
        m = pattern.match(line.strip())
        if m:
            results[m.group(1)] = float(m.group(2))
    return results


def parse_summary(path):
    """Parse summary.md table into {problem: {baseline: avg}}."""
    results = {p: {} for p in PROBLEM_ORDER}
    pattern = re.compile(r"^\|\s*(\w+)\s*\|(.+)\|$")
    for line in Path(path).read_text().splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        name = m.group(1)
        if name not in results:
            continue
        values = [v.strip() for v in m.group(2).split("|")]
        if len(values) != len(BASELINES):
            continue
        for bl, v in zip(BASELINES, values):
            try:
                results[name][bl] = float(v)
            except ValueError:
                pass
    return results


def main():
    solution = parse_solution(base_dir / "Solution.txt")
    baselines = parse_summary(base_dir / "baseline" / "summary.md")

    missing = [p for p in PROBLEM_ORDER if p not in solution]
    if missing:
        print(f"Warning: missing from Solution.txt: {missing}\n")

    col_w = 14
    header = f"{'Problem':<12}" + "".join(f"{bl:>{col_w}}" for bl in BASELINES)
    print(header)
    print("-" * len(header))

    wins = {bl: 0 for bl in BASELINES}
    total = {bl: 0 for bl in BASELINES}

    for prob in PROBLEM_ORDER:
        if prob not in solution:
            continue
        my_score = solution[prob]
        row = f"{prob:<12}"
        for bl in BASELINES:
            bl_score = baselines.get(prob, {}).get(bl)
            if bl_score is None:
                row += f"{'N/A':>{col_w}}"
                continue
            diff = my_score - bl_score
            row += f"{diff:>+{col_w}.1f}"
            total[bl] += 1
            if diff > 0:
                wins[bl] += 1
        print(row)

    print("-" * len(header))
    win_row = f"{'Win %':<12}"
    for bl in BASELINES:
        pct = 100 * wins[bl] / total[bl] if total[bl] else 0
        win_row += f"{pct:>{col_w - 1}.1f}%"
    print(win_row)

    win_count_row = f"{'Wins':<12}"
    for bl in BASELINES:
        win_count_row += f"{wins[bl]:>{col_w - 1}d}/{total[bl]}"
    print(win_count_row)


if __name__ == "__main__":
    main()
