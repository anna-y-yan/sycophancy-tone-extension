"""Write synthetic result JSONL for testing the plot without running inference."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TONES = [
    "casual_conversational",
    "unconfident_academic",
    "confident_academic",
    "authoritative_directive",
    "high_stakes_urgent",
]
# Synthetic rates: (sycophancy when user said wrong answer, accuracy)
# High-stakes slightly more sycophantic, authoritative less, etc.
DEMO_RATES = {
    "casual_conversational": (0.35, 0.75),
    "unconfident_academic": (0.45, 0.70),
    "confident_academic": (0.25, 0.82),
    "authoritative_directive": (0.20, 0.85),
    "high_stakes_urgent": (0.55, 0.65),
}
N_PER_CELL = 20


def main():
    out = ROOT / "results" / "demo_results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in ["gpt-4o-mini", "claude-3-5-sonnet-20241022"]:
        for tone in TONES:
            syc_rate, acc_rate = DEMO_RATES[tone]
            for i in range(N_PER_CELL):
                rows.append({
                    "model": model,
                    "tone": tone,
                    "condition": "think_incorrect",
                    "task": "answer",
                    "base": {},
                    "response": "",
                    "sycophantic": 1 if i < int(N_PER_CELL * syc_rate) else 0,
                    "correct": 1 if i < int(N_PER_CELL * acc_rate) else 0,
                })
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} demo rows to {out}")


if __name__ == "__main__":
    main()
