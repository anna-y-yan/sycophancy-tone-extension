"""
Produce a figure from inference results: sycophancy rate and accuracy by prompt tone (and by model).

Usage:
  python analysis/plot_results.py --results results/results_answer_*.jsonl --out results_figure.png
  python analysis/plot_results.py --summary summary_by_tone.json --out results_figure.png
"""

import argparse
import json
from pathlib import Path


# Display order and short labels for tones
TONE_ORDER = [
    "casual_conversational",
    "unconfident_academic",
    "confident_academic",
    "authoritative_directive",
    "high_stakes_urgent",
]
TONE_LABELS = {
    "casual_conversational": "Casual",
    "unconfident_academic": "Unconfident\nacademic",
    "confident_academic": "Confident\nacademic",
    "authoritative_directive": "Authoritative",
    "high_stakes_urgent": "High-stakes",
}


def load_results(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
    return out


def aggregate_for_plot(results: list[dict]) -> tuple[dict, list[str]]:
    """Returns (by_model_tone: {(model, tone): {sycophancy_rate, accuracy, n}}, models)."""
    from collections import defaultdict
    by = defaultdict(lambda: {"n": 0, "sycophantic": 0, "correct": 0})
    models = set()
    for r in results:
        model = r.get("model", "unknown")
        tone = r.get("tone", "unknown")
        models.add(model)
        key = (model, tone)
        by[key]["n"] += 1
        if r.get("sycophantic"):
            by[key]["sycophantic"] += 1
        if r.get("correct"):
            by[key]["correct"] += 1
    for k in by:
        d = by[k]
        n = d["n"]
        d["sycophancy_rate"] = d["sycophantic"] / n if n else 0
        d["accuracy"] = d["correct"] / n if n else 0
    return dict(by), sorted(models)


def plot_from_results(results: list[dict], out_path: Path, condition_filter: str | None = "think_incorrect", use_html: bool = False):
    """
    condition_filter: only include this condition when computing rates (e.g. think_incorrect for sycophancy).
    If None, use all rows per (model, tone).
    """
    if condition_filter:
        results = [r for r in results if r.get("condition") == condition_filter]
    if not results:
        raise SystemExit("No results to plot (try without --condition-filter).")
    by, models = aggregate_for_plot(results)
    plot_bars(by, models, out_path, use_html_fallback=use_html)


def _model_label(m: str) -> str:
    return m.replace("claude-3-5-sonnet-20241022", "Claude 3.5").replace("gpt-4o-mini", "GPT-4o-mini")


def plot_bars_html(by: dict, models: list, out_path: Path):
    """Write HTML with CSS bar charts (no matplotlib)."""
    tones = [t for t in TONE_ORDER if any((m, t) in by for m in models)]
    if not tones:
        tones = sorted(set(t for _, t in by.keys()))
    max_w = 180
    model_colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"][: len(models)]
    rows = []
    for title, metric in [("Sycophancy by prompt tone", "sycophancy_rate"), ("Accuracy by prompt tone", "accuracy")]:
        rows.append(f"<h2>{title}</h2>")
        rows.append('<div class="chart">')
        for tone in tones:
            label = TONE_LABELS.get(tone, tone.replace("_", " "))
            rows.append(f'<div class="tone-row"><span class="tone-label">{label}</span>')
            for i, model in enumerate(models):
                val = by.get((model, tone), {}).get(metric, 0)
                pct = int(round(val * 100))
                color = model_colors[i % len(model_colors)]
                rows.append(f'<span class="bar-wrap" title="{_model_label(model)}: {pct}%"><span class="bar" style="width:{val*max_w}px;background:{color}" data-val="{val:.2f}"></span><span class="bar-pct">{pct}%</span></span>')
            rows.append("</div>")
        legend = "".join(f'<span class="leg"><span class="leg-dot" style="background:{model_colors[i]}"></span>{_model_label(m)}</span>' for i, m in enumerate(models))
        rows.append(f'<div class="legend">{legend}</div>')
        rows.append("</div>")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sycophancy by tone</title>
<style>
body {{ font-family: sans-serif; margin: 24px; max-width: 720px; }}
h1 {{ font-size: 1.25rem; }}
h2 {{ font-size: 1rem; margin-top: 24px; margin-bottom: 8px; }}
.chart {{ margin-bottom: 16px; }}
.tone-row {{ display: flex; align-items: center; margin: 4px 0; gap: 8px; }}
.tone-label {{ width: 120px; font-size: 11px; flex-shrink: 0; }}
.bar-wrap {{ display: inline-flex; align-items: center; }}
.bar {{ height: 18px; min-width: 2px; border-radius: 2px; }}
.bar-pct {{ font-size: 10px; margin-left: 4px; width: 28px; }}
.legend {{ margin-top: 6px; font-size: 11px; color: #444; display: flex; gap: 12px; flex-wrap: wrap; }}
.leg-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }}
</style></head><body>
<h1>Prompt tone & sycophancy</h1>
<p>Sycophancy rate = % of trials (user stated wrong answer) where model agreed with user. Accuracy = % correct.</p>
{"".join(rows)}
</body></html>"""
    out_path = out_path.with_suffix(".html") if out_path.suffix != ".html" else out_path
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved figure to {out_path}")


def plot_bars(by: dict, models: list, out_path: Path, use_html_fallback: bool = False):
    """Draw figure from aggregated (model, tone) -> {sycophancy_rate, accuracy, n}."""
    if use_html_fallback or out_path.suffix.lower() == ".html":
        plot_bars_html(by, models, out_path)
        return
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Matplotlib unavailable ({e}), writing HTML figure instead.")
        plot_bars_html(by, models, out_path)
        return
    tones = [t for t in TONE_ORDER if any((m, t) in by for m in models)]
    if not tones:
        tones = sorted(set(t for _, t in by.keys()))
    x = np.arange(len(tones))
    width = 0.8 / len(models) if len(models) > 1 else 0.6
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, ylabel, title in [
        (axes[0], "sycophancy_rate", "Sycophancy rate", "Sycophancy by prompt tone"),
        (axes[1], "accuracy", "Accuracy", "Accuracy by prompt tone"),
    ]:
        for i, model in enumerate(models):
            offset = (i - (len(models) - 1) / 2) * width
            vals = [by.get((model, t), {}).get(metric, 0) for t in tones]
            ax.bar(x + offset, vals, width, label=_model_label(model))
        ax.set_xticks(x)
        ax.set_xticklabels([TONE_LABELS.get(t, t.replace("_", "\n")) for t in tones], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {out_path}")


def plot_from_summary(summary_path: Path, out_path: Path, format: str = "auto"):
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    by_mt = summary.get("by_model_and_tone", {})
    if not by_mt:
        raise SystemExit("Summary has no 'by_model_and_tone'; run aggregate first with result files that include 'model'.")
    by = {}
    models = set()
    for key, d in by_mt.items():
        if "|" in key:
            model, tone = key.split("|", 1)
            models.add(model)
            by[(model, tone)] = d
    use_html = format == "html" or (format == "auto" and out_path.suffix.lower() == ".html")
    plot_bars(by, sorted(models), out_path, use_html_fallback=use_html)


def main():
    parser = argparse.ArgumentParser(description="Plot sycophancy and accuracy by tone from results or summary.")
    parser.add_argument("--results", type=Path, nargs="+", default=None, help="Result JSONL file(s)")
    parser.add_argument("--summary", type=Path, default=None, help="Summary JSON from aggregate_by_tone.py")
    parser.add_argument("--out", type=Path, default=Path("results_figure.png"))
    parser.add_argument("--format", choices=["png", "html", "auto"], default="auto", help="Output format; auto picks from --out extension or tries PNG then HTML.")
    parser.add_argument(
        "--condition-filter",
        default="think_incorrect",
        help="Only include this condition for rates (default: think_incorrect). Use 'none' to include all.",
    )
    args = parser.parse_args()
    if args.summary:
        if args.results:
            raise SystemExit("Use either --results or --summary, not both.")
        plot_from_summary(args.summary, args.out, format=args.format)
    elif args.results:
        cf = None if (args.condition_filter or "").lower() == "none" else args.condition_filter
        results = load_results(args.results)
        print(f"Loaded {len(results)} rows from {len(args.results)} file(s)")
        plot_from_results(results, args.out, condition_filter=cf, use_html=(args.format == "html"))
    else:
        raise SystemExit("Provide either --results or --summary.")


if __name__ == "__main__":
    main()
