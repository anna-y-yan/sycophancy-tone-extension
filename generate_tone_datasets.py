#!/usr/bin/env python3
"""
Generate tone-variant SycophancyEval datasets from the original JSONL files.

Usage:
  python generate_tone_datasets.py --task answer --datasets-dir /path/to/sycophancy-eval/datasets --out-dir ./datasets_by_tone
  python generate_tone_datasets.py --task feedback --task all ...
"""

import argparse
import json
import os
from pathlib import Path

from tone_templates import (
    ANSWER_TONE_TEMPLATES,
    FEEDBACK_CONDITION_FROM_TEMPLATE_TYPE,
    FEEDBACK_TONE_TEMPLATES,
    TONES,
)


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Error in line {i+1} of {path}: {e}")
    return out


def detect_answer_condition(metadata: dict, base: dict) -> str:
    """Detect condition from original SycophancyEval prompt_template (unfilled, with placeholders)."""
    template = metadata.get("prompt_template") or ""
    if not template.strip():
        return "no_belief"
    # Original templates use {question}, {correct_answer}, {incorrect_answer}
    if "I don't think the answer is {correct_answer}" in template:
        return "doubt_correct"
    if "I think the answer is {incorrect_answer}" in template:
        return "think_incorrect"
    if "I think the answer is {correct_answer}" in template:
        return "think_correct"
    return "no_belief"


def build_answer_prompt_content(template_str: str, base: dict) -> str:
    return template_str.format(
        question=base["question"],
        correct_answer=base["correct_answer"],
        incorrect_answer=base["incorrect_answer"],
    )


def generate_answer_tone_datasets(datasets_dir: Path, out_dir: Path) -> None:
    path = datasets_dir / "answer.jsonl"
    if not path.exists():
        print(f"Skip answer: {path} not found")
        return
    rows = load_jsonl(path)
    print(f"Loaded {len(rows)} rows from answer.jsonl")
    for tone in TONES:
        templates = ANSWER_TONE_TEMPLATES.get(tone)
        if not templates:
            continue
        out_path = out_dir / "answer" / f"{tone}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for item in rows:
                base = item["base"]
                metadata = item.get("metadata", {})
                cond = detect_answer_condition(metadata, base)
                template_str = templates.get(cond)
                if not template_str:
                    continue
                try:
                    content = build_answer_prompt_content(template_str, base)
                except KeyError:
                    continue
                new_item = {
                    "prompt": [{"type": "human", "content": content}],
                    "base": base,
                    "metadata": {**metadata, "tone": tone, "condition": cond, "prompt_template": template_str},
                }
                f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                count += 1
        print(f"  Wrote {count} rows to {out_path}")


def generate_feedback_tone_datasets(datasets_dir: Path, out_dir: Path) -> None:
    path = datasets_dir / "feedback.jsonl"
    if not path.exists():
        print(f"Skip feedback: {path} not found")
        return
    rows = load_jsonl(path)
    print(f"Loaded {len(rows)} rows from feedback.jsonl")
    for tone in TONES:
        templates = FEEDBACK_TONE_TEMPLATES.get(tone)
        if not templates:
            continue
        out_path = out_dir / "feedback" / f"{tone}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for item in rows:
                base = item["base"]
                metadata = item.get("metadata", {})
                template_type = metadata.get("prompt_template_type", "")
                cond = FEEDBACK_CONDITION_FROM_TEMPLATE_TYPE.get(template_type, "neutral")
                template_str = templates.get(cond)
                if not template_str:
                    continue
                text = base.get("text", "")
                try:
                    content = template_str.format(text=text)
                except KeyError:
                    content = template_str.replace("{text}", text)
                new_item = {
                    "prompt": [{"type": "human", "content": content}],
                    "base": base,
                    "metadata": {**metadata, "tone": tone, "condition": cond, "prompt_template": template_str},
                }
                f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                count += 1
        print(f"  Wrote {count} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate tone-variant sycophancy eval datasets")
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path(os.environ.get("SYCOPHANCY_EVAL_DATASETS", "../sycophancy-eval/datasets")),
        help="Path to sycophancy-eval datasets folder (answer.jsonl, feedback.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./datasets_by_tone"),
        help="Output directory; creates task/tone.jsonl under it",
    )
    parser.add_argument(
        "--task",
        choices=["answer", "feedback", "all"],
        default="answer",
        help="Which task(s) to generate",
    )
    args = parser.parse_args()
    if not args.datasets_dir.is_dir():
        print(f"Datasets dir not found: {args.datasets_dir}")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.task in ("answer", "all"):
        generate_answer_tone_datasets(args.datasets_dir, args.out_dir)
    if args.task in ("feedback", "all"):
        generate_feedback_tone_datasets(args.datasets_dir, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
