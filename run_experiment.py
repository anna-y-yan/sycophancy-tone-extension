#!/usr/bin/env python3
"""
End-to-end: generate datasets (if needed), run inference, aggregate, and plot.

Usage:
  # Full run (needs OPENAI_API_KEY and/or ANTHROPIC_API_KEY)
  python run_experiment.py --models gpt-4o-mini,claude-3-5-sonnet-20241022

  # Quick test (few examples per tone)
  python run_experiment.py --models gpt-4o-mini --limit 2

  # Plot only (from existing results)
  python run_experiment.py --plot-only --results results/results_answer_*.jsonl
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", type=Path, default=ROOT / "sample_data", help="Original datasets (answer.jsonl)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "datasets_by_tone")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--models", default="gpt-4o-mini", help="Comma-separated model names")
    parser.add_argument("--limit", type=int, default=None, help="Max rows per tone (for quick test)")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate tone datasets from --datasets-dir even if datasets_by_tone already exists (use when switching from sample to full data)")
    parser.add_argument("--base-url", default=None, help="Local OpenAI-compatible API URL (e.g. http://localhost:11434/v1 for Ollama). No paid API key needed.")
    parser.add_argument("--plot-only", action="store_true", help="Skip inference; plot from existing results")
    parser.add_argument("--results", type=Path, nargs="*", help="Result JSONL files for --plot-only (e.g. results/*.jsonl)")
    parser.add_argument("--figure", type=Path, default=ROOT / "results" / "sycophancy_by_tone.html", help="Output figure (use .html to avoid matplotlib)")
    args = parser.parse_args()

    if args.plot_only:
        result_files = args.results or list(args.results_dir.glob("*.jsonl"))
        if not result_files:
            print("No result files found. Run without --plot-only first, or pass --results path1 path2 ...")
            sys.exit(1)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        fmt = "html" if args.figure.suffix.lower() == ".html" else "auto"
        subprocess.run(
            [sys.executable, str(ROOT / "analysis" / "plot_results.py"), "--results"] + [str(p) for p in result_files] + ["--out", str(args.figure), "--format", fmt],
            check=True,
            cwd=ROOT,
        )
        print(f"Figure saved to {args.figure}")
        return

    # 1. Generate tone datasets if missing or --regenerate
    need_generate = args.regenerate or not (args.out_dir / "answer").exists() or not list((args.out_dir / "answer").glob("*.jsonl"))
    if need_generate:
        if args.regenerate and (args.out_dir / "answer").exists():
            for f in (args.out_dir / "answer").glob("*.jsonl"):
                f.unlink()
        print("Generating tone datasets...")
        subprocess.run(
            [sys.executable, str(ROOT / "generate_tone_datasets.py"), "--task", "answer", "--datasets-dir", str(args.datasets_dir), "--out-dir", str(args.out_dir)],
            check=True,
            cwd=ROOT,
        )
    else:
        print("Tone datasets already present.")

    # 2. Run inference
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "run_inference.py"),
        "--datasets-dir", str(args.out_dir),
        "--task", "answer",
        "--models", args.models,
        "--out", str(args.results_dir / "results.jsonl"),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    print("Running inference (requires API key(s))...")
    subprocess.run(cmd, check=True, cwd=ROOT)

    # 3. Aggregate (only inference outputs, not demo_results.jsonl)
    result_files = [
        f for f in args.results_dir.glob("*.jsonl")
        if f.name != "demo_results.jsonl"
    ]
    if not result_files:
        print("No result files produced.")
        sys.exit(1)
    summary_path = args.results_dir / "summary_by_tone.json"
    subprocess.run(
        [sys.executable, str(ROOT / "analysis" / "aggregate_by_tone.py"), "--results"] + [str(p) for p in result_files] + ["--out", str(summary_path)],
        check=True,
        cwd=ROOT,
    )

    # 4. Plot (use .html to avoid matplotlib; use .png if you have matplotlib)
    fmt = "html" if args.figure.suffix.lower() == ".html" else "auto"
    subprocess.run(
        [sys.executable, str(ROOT / "analysis" / "plot_results.py"), "--summary", str(summary_path), "--out", str(args.figure), "--format", fmt],
        check=True,
        cwd=ROOT,
    )
    print(f"Done. Figure: {args.figure}")


if __name__ == "__main__":
    main()
