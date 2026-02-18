"""
Aggregate inference results by tone for sycophancy analysis.

Expects a results JSONL where each line has at least:
  - tone: str (e.g. casual_conversational, high_stakes_urgent)
  - condition: str (e.g. think_incorrect, think_correct, like, dislike)
  - sycophantic: bool (True if model agreed with user over correct answer)
  - correct: bool (True if model gave the correct answer)

Optional: task, model_name, response.

Usage:
  python aggregate_by_tone.py --results results.jsonl --out summary_by_tone.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_results(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def aggregate(results: list[dict]) -> dict:
    by_tone = defaultdict(lambda: {"n": 0, "sycophantic": 0, "correct": 0})
    by_tone_and_condition = defaultdict(lambda: {"n": 0, "sycophantic": 0, "correct": 0})
    by_model_and_tone = defaultdict(lambda: {"n": 0, "sycophantic": 0, "correct": 0})
    for r in results:
        tone = r.get("tone", "unknown")
        cond = r.get("condition", "unknown")
        model = r.get("model", "unknown")
        by_tone[tone]["n"] += 1
        if r.get("sycophantic"):
            by_tone[tone]["sycophantic"] += 1
        if r.get("correct"):
            by_tone[tone]["correct"] += 1
        key = (tone, cond)
        by_tone_and_condition[key]["n"] += 1
        if r.get("sycophantic"):
            by_tone_and_condition[key]["sycophantic"] += 1
        if r.get("correct"):
            by_tone_and_condition[key]["correct"] += 1
        mkey = (model, tone)
        by_model_and_tone[mkey]["n"] += 1
        if r.get("sycophantic"):
            by_model_and_tone[mkey]["sycophantic"] += 1
        if r.get("correct"):
            by_model_and_tone[mkey]["correct"] += 1
    # Rates
    def rates(d):
        n = d["n"]
        return {
            **d,
            "sycophancy_rate": d["sycophantic"] / n if n else 0,
            "accuracy": d["correct"] / n if n else 0,
        }
    summary = {
        "by_tone": {t: rates(d) for t, d in by_tone.items()},
        "by_tone_and_condition": {
            f"{t}|{c}": rates(d) for (t, c), d in by_tone_and_condition.items()
        },
        "by_model_and_tone": {
            f"{m}|{t}": rates(d) for (m, t), d in by_model_and_tone.items()
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="One or more JSONL result files (e.g. from OpenAI and Claude)")
    parser.add_argument("--out", type=Path, default=Path("summary_by_tone.json"), help="Output JSON summary")
    args = parser.parse_args()
    results = []
    for path in args.results:
        results.extend(load_results(path))
    summary = aggregate(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Processed {len(results)} results -> {args.out}")
    if summary.get("by_model_and_tone"):
        print("By model and tone:")
        for key, d in sorted(summary["by_model_and_tone"].items()):
            print(f"  {key}: n={d['n']} sycophancy_rate={d['sycophancy_rate']:.2%} accuracy={d['accuracy']:.2%}")
    else:
        for tone, d in summary["by_tone"].items():
            print(f"  {tone}: n={d['n']} sycophancy_rate={d['sycophancy_rate']:.2%} accuracy={d['accuracy']:.2%}")


if __name__ == "__main__":
    main()
