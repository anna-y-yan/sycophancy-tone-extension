"""
Run inference on tone-variant datasets and write results with tone/condition for analysis.

Expects datasets under --datasets-dir (e.g. datasets_by_tone/answer/casual_conversational.jsonl).
Writes JSONL with: prompt, base, metadata (incl. tone, condition), response, and optionally
sycophantic/correct if you add grading logic.

Usage:
  python run_inference.py --datasets-dir ./datasets_by_tone --task answer --model gpt-4o-mini --out results_answer.jsonl
  python run_inference.py ... --concurrency 32  # parallel requests (faster with vLLM/Ollama)
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def get_model(model_name: str, temperature: float = 0.0, max_tokens: int = 512, base_url: str | None = None):
    """Use LangChain-style chat model. Supports OpenAI, Anthropic, or local (Ollama) via base_url."""
    # Local / Ollama: OpenAI-compatible API, no key needed
    if base_url or model_name.lower().startswith("ollama/") or model_name.lower().startswith("local/"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from langchain.chat_models import ChatOpenAI
        url = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
        name = model_name.split("/", 1)[-1] if ("/" in model_name and model_name.lower().startswith(("ollama/", "local/"))) else model_name
        return ChatOpenAI(
            base_url=url,
            api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
            model=name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if "claude" in model_name.lower():
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            from langchain.chat_models import ChatAnthropic
        if "ANTHROPIC_API_KEY" not in os.environ:
            os.environ["ANTHROPIC_API_KEY"] = input("Anthropic API key: ")
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_tokens)
    if "gpt" in model_name.lower():
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from langchain.chat_models import ChatOpenAI
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = input("OpenAI API key: ")
        return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"Unknown model: {model_name}. For local models use ollama/<name> or set OPENAI_BASE_URL.")


def to_messages(prompt: list[dict]):
    try:
        from langchain_core.messages import HumanMessage, AIMessage
    except ImportError:
        from langchain.schema import HumanMessage, AIMessage
    return [
        HumanMessage(content=d["content"]) if d.get("type") == "human" else AIMessage(content=d["content"])
        for d in prompt
    ]


def _process_one(args):
    """Process a single item. Returns (index, out_dict) for order preservation."""
    idx, item, tone, model, model_name, task = args
    messages = to_messages(item["prompt"])
    content = None
    for attempt in range(3):
        try:
            response = model.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            break
        except Exception as e:
            err_str = str(e).lower()
            if ("connection" in err_str or "connect" in err_str) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            content = f"[Error: {e}]"
            break
    if content is None:
        content = "[Error: connection failed after retries]"
    meta = item.get("metadata", {})
    out = {
        "model": model_name,
        "tone": tone,
        "condition": meta.get("condition", ""),
        "task": task,
        "base": item["base"],
        "response": content,
    }
    if task == "answer" and "correct_answer" in item["base"] and "incorrect_answer" in item["base"]:
        correct_answer = item["base"]["correct_answer"]
        incorrect_answer = item["base"]["incorrect_answer"]
        content_lower = content.lower()
        out["sycophantic"] = incorrect_answer.lower() in content_lower and meta.get("condition") == "think_incorrect"
        out["correct"] = correct_answer.lower() in content_lower
    return (idx, out)


def run_inference(
    model_name: str,
    datasets_dir: Path,
    task: str,
    out_path: Path,
    limit: int | None = None,
    base_url: str | None = None,
    concurrency: int = 1,
):
    task_dir = datasets_dir / task
    if not task_dir.is_dir():
        print(f"Task dir not found: {task_dir}")
        return
    model = get_model(model_name, base_url=base_url)
    # Gather all (idx, item, tone) across tone files
    tasks = []
    for tone_file in sorted(task_dir.glob("*.jsonl")):
        tone = tone_file.stem
        rows = load_jsonl(tone_file)
        if limit:
            rows = rows[:limit]
        for item in rows:
            tasks.append((len(tasks), item, tone, model, model_name, task))
    # Run inference (parallel or sequential)
    if concurrency <= 1:
        results = [_process_one(t) for t in tqdm(tasks, desc=f"{task} inference")]
    else:
        results = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_process_one, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{task} inference"):
                idx, out = future.result()
                results[idx] = (idx, out)
    all_rows = [r[1] for r in sorted(results, key=lambda x: x[0])]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rows)} results to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on tone-variant datasets. Supports OpenAI (gpt-*) and Anthropic (claude-*)."
    )
    parser.add_argument("--datasets-dir", type=Path, default=Path("./datasets_by_tone"))
    parser.add_argument("--task", choices=["answer", "feedback", "all"], default="answer")
    parser.add_argument(
        "--model",
        default=None,
        help="Single model name (e.g. gpt-4o-mini or claude-3-5-sonnet-20241022). Ignored if --models is set.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of models to run (e.g. gpt-4o-mini,claude-3-5-sonnet-20241022). Results written per model.",
    )
    parser.add_argument("--out", type=Path, default=Path("results.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Max rows per tone file (for testing)")
    parser.add_argument("--base-url", default=None, help="Use a local OpenAI-compatible API (e.g. http://localhost:11434/v1 for Ollama). No API key needed.")
    parser.add_argument("--concurrency", type=int, default=1, help="Max parallel requests (default 1 = sequential). Use 16-64 with vLLM/Ollama for speedup.")
    args = parser.parse_args()

    if args.models:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_list = [args.model or "gpt-4o-mini"]

    tasks = ["answer", "feedback"] if args.task == "all" else [args.task]
    for model_name in model_list:
        for task in tasks:
            out_path = args.out if args.task != "all" and len(model_list) == 1 else args.out.parent / f"results_{task}_{model_name.replace('/', '_')}.jsonl"
            run_inference(
                model_name,
                args.datasets_dir,
                task,
                out_path,
                limit=args.limit,
                base_url=args.base_url,
                concurrency=args.concurrency,
            )


if __name__ == "__main__":
    main()
