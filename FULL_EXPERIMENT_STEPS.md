# Steps to Run the Full Experiment

Follow these in order. All commands are run from the **sycophancy-tone-extension** directory unless noted.

---

## 1. Get the full SycophancyEval datasets

Clone the original repo (sibling to this one or anywhere you like):

```bash
cd /path/to/parent
git clone https://github.com/meg-tong/sycophancy-eval.git
```

Remember the path to its `datasets/` folder, e.g.:
`/path/to/parent/sycophancy-eval/datasets`

---

## 2. Python environment and dependencies

```bash
cd /Users/annayan/sycophancy-tone-extension
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install langchain-openai langchain-anthropic tqdm
```

---

## 3. API keys (for OpenAI and/or Claude)

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...   # optional, only if running Claude
```

---

## 4. Generate tone datasets from the full data

**Important:** Use the path to **meg-tong’s** `datasets/` folder so you get the full `answer.jsonl` (and optionally `feedback.jsonl`), not the 3-row sample.

```bash
# Answer task only (TriviaQA-style; this is what the one-shot experiment uses)
python generate_tone_datasets.py --task answer \
  --datasets-dir /path/to/sycophancy-eval/datasets \
  --out-dir ./datasets_by_tone

# Optional: also generate feedback task
python generate_tone_datasets.py --task feedback \
  --datasets-dir /path/to/sycophancy-eval/datasets \
  --out-dir ./datasets_by_tone

# Or both at once
python generate_tone_datasets.py --task all \
  --datasets-dir /path/to/sycophancy-eval/datasets \
  --out-dir ./datasets_by_tone
```

Output: `datasets_by_tone/answer/casual_conversational.jsonl`, …, and optionally `datasets_by_tone/feedback/...`.

---

## 5. Run inference

Uses the **tone** datasets you just generated (`--datasets-dir` = `./datasets_by_tone`).

**Single model (e.g. OpenAI):**
```bash
python run_inference.py --datasets-dir ./datasets_by_tone --task answer \
  --model gpt-4o-mini \
  --out results/results.jsonl
```

**Multiple models (OpenAI + Claude):**
```bash
python run_inference.py --datasets-dir ./datasets_by_tone --task answer \
  --models gpt-4o-mini,claude-3-5-sonnet-20241022 \
  --out results/results.jsonl
```
This writes `results/results_answer_gpt-4o-mini.jsonl` and `results/results_answer_claude-3-5-sonnet-20241022.jsonl`.

**Quick test (few rows per tone):**
```bash
python run_inference.py --datasets-dir ./datasets_by_tone --task answer \
  --model gpt-4o-mini --out results/results.jsonl --limit 5
```

---

## 6. Aggregate results by tone

```bash
# One result file
python analysis/aggregate_by_tone.py --results results/results_answer_gpt-4o-mini.jsonl --out results/summary_by_tone.json

# Multiple (e.g. OpenAI + Claude)
python analysis/aggregate_by_tone.py \
  --results results/results_answer_gpt-4o-mini.jsonl results/results_answer_claude-3-5-sonnet-20241022.jsonl \
  --out results/summary_by_tone.json
```

---

## 7. Plot the figure

```bash
python analysis/plot_results.py --summary results/summary_by_tone.json \
  --out results/sycophancy_by_tone.html --format html
```

Open `results/sycophancy_by_tone.html` in a browser.

---

## One-shot: full pipeline with one command

If you **already generated** full tone datasets (step 4) into `./datasets_by_tone`, you can run generation + inference + aggregate + plot in one go. You **must** pass the **original** meg-tong datasets dir so it doesn’t use the small sample:

**Option A – Regenerate from full data and run:**  
Delete or rename existing `datasets_by_tone`, then:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

python run_experiment.py \
  --datasets-dir /path/to/sycophancy-eval/datasets \
  --models gpt-4o-mini,claude-3-5-sonnet-20241022
```

**If you already have tone files** (e.g. from the 3-row sample), the script skips generation and reuses them. To use the full meg-tong data instead, clear the tone datasets first, then run the command above:

```bash
rm -rf datasets_by_tone/answer/*.jsonl
python run_experiment.py --datasets-dir /path/to/sycophancy-eval/datasets \
  --models gpt-4o-mini,claude-3-5-sonnet-20241022
```

Output: `results/summary_by_tone.json` and `results/sycophancy_by_tone.html`.
</think>
Simplifying the one-shot section and creating the steps doc.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
StrReplace