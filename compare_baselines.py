import re
import sys
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
    """Parse a Solution*.txt file, keeping the last value for each problem."""
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


def find_solution_files(explicit_args):
    """Return ordered list of (label, path) for solution files to compare."""
    if explicit_args:
        return [(Path(p).stem, base_dir / p) for p in explicit_args]
    # Auto-detect: Solution.txt + any Solution_v*.txt
    files = sorted(base_dir.glob("Solution_v*.txt"))
    default = base_dir / "Solution.txt"
    candidates = ([default] if default.exists() else []) + files
    return [(p.stem, p) for p in candidates]


def main():
    sol_files = find_solution_files(sys.argv[1:])
    if not sol_files:
        print("No solution files found.")
        return

    solutions = {}
    for label, path in sol_files:
        if not path.exists():
            print(f"Warning: {path} not found, skipping.")
            continue
        solutions[label] = parse_solution(path)

    if not solutions:
        return

    baselines = parse_summary(base_dir / "baseline" / "summary.md")

    col_w = 13
    sol_labels = list(solutions.keys())

    # ── Header ────────────────────────────────────────────────────────────
    sep = "  |  "
    bl_header  = "".join(f"{bl:>{col_w}}" for bl in BASELINES)
    sol_header = "".join(f"{lb:>{col_w}}" for lb in sol_labels)
    header = f"{'Problem':<12}{bl_header}{sep}{sol_header}"
    divider = "-" * len(header)
    print(header)
    print(divider)

    wins  = {bl: {lb: 0 for lb in sol_labels} for bl in BASELINES}
    total = {bl: 0 for bl in BASELINES}

    for prob in PROBLEM_ORDER:
        # Skip if no solution has this problem yet
        if not any(prob in s for s in solutions.values()):
            continue

        # Baseline diffs (relative to first solution with data)
        primary = next((s[prob] for s in solutions.values() if prob in s), None)
        bl_part = ""
        for bl in BASELINES:
            bl_score = baselines.get(prob, {}).get(bl)
            if bl_score is None or primary is None:
                bl_part += f"{'N/A':>{col_w}}"
                continue
            diff = primary - bl_score
            bl_part += f"{diff:>+{col_w}.1f}"
            total[bl] += 1
            for lb, sol in solutions.items():
                if prob in sol and sol[prob] > bl_score:
                    wins[bl][lb] += 1

        # Absolute scores for each solution file
        sol_part = ""
        scores = [solutions[lb].get(prob) for lb in sol_labels]
        best   = max((s for s in scores if s is not None), default=None)
        for score in scores:
            if score is None:
                sol_part += f"{'---':>{col_w}}"
            elif len(sol_labels) > 1 and score == best:
                sol_part += f"{'*'+f'{score:.1f}':>{col_w}}"
            else:
                sol_part += f"{score:>{col_w}.1f}"

        print(f"{prob:<12}{bl_part}{sep}{sol_part}")

    # ── Sum row ───────────────────────────────────────────────────────────
    print(divider)
    sums = {lb: sum(solutions[lb].get(p, 0) for p in PROBLEM_ORDER) for lb in sol_labels}
    bl_sums = {bl: sum(baselines.get(p, {}).get(bl, 0) for p in PROBLEM_ORDER) for bl in BASELINES}
    best_sum = max(sums.values()) if sums else None
    sum_bl = "".join(f"{sums[next(iter(sol_labels))] - bl_sums[bl]:>+{col_w}.1f}" for bl in BASELINES)
    sum_sol = ""
    for lb in sol_labels:
        s = sums[lb]
        tag = "*" if len(sol_labels) > 1 and s == best_sum else ""
        sum_sol += f"{tag+f'{s:.1f}':>{col_w}}"
    print(f"{'SUM':<12}{sum_bl}{sep}{sum_sol}")
    print(divider)

    # ── Win % rows ────────────────────────────────────────────────────────
    for lb in sol_labels:
        win_row = f"{'Win% '+lb:<12}"
        for bl in BASELINES:
            n = total[bl]
            w = wins[bl][lb]
            pct = 100 * w / n if n else 0
            win_row += f"{pct:>{col_w - 5}.1f}%({w}/{n})"
        win_row += sep
        for lb2 in sol_labels:
            win_row += f"{'':>{col_w}}"
        print(win_row)

    # ── Head-to-head between solution files ───────────────────────────────
    if len(sol_labels) >= 2:
        print()
        print("Head-to-head (problems where both have results):")
        for i, lb_a in enumerate(sol_labels):
            for lb_b in sol_labels[i + 1:]:
                w_a = w_b = ties = 0
                for prob in PROBLEM_ORDER:
                    a = solutions[lb_a].get(prob)
                    b = solutions[lb_b].get(prob)
                    if a is None or b is None:
                        continue
                    if a > b:
                        w_a += 1
                    elif b > a:
                        w_b += 1
                    else:
                        ties += 1
                n = w_a + w_b + ties
                print(f"  {lb_a} vs {lb_b}:  "
                      f"{lb_a} wins {w_a}/{n}  |  "
                      f"{lb_b} wins {w_b}/{n}  |  "
                      f"ties {ties}/{n}")


if __name__ == "__main__":
    main()
