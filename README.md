# Prompt Tone and Sycophancy Extension

Research extension for studying **how prompt tone affects sycophancy in LLMs**, building on [meg-tong/sycophancy-eval](https://github.com/meg-tong/sycophancy-eval) (Sharma et al., "Towards Understanding Sycophancy in Language Models," ICLR 2024).

## Research question

Does the **tone** of the user prompt (casual vs. academic vs. authoritative vs. urgent) enhance or diminish sycophantic behavior—i.e., the model agreeing with the user’s stated belief instead of giving the correct answer?

## Tone categories

1. **Casual / conversational** – informal, friendly, “just wondering”
2. **Unconfident academic** – hedging, “I’m not sure,” “I might be wrong”
3. **Confident academic** – assured, scholarly framing
4. **Authoritative / directive** – commanding, expert-like, “tell me,” “you must”
5. **High-stakes / urgent** – time pressure, importance, “I need this correct”

## Setup

1. **Clone the original sycophancy-eval repo** (if you haven’t):

   ```bash
   git clone https://github.com/meg-tong/sycophancy-eval.git
   ```

2. **Install dependencies** (from `sycophancy-eval` or your venv):

   ```bash
   pip install langchain-openai langchain-anthropic tqdm
   ```

3. **Point this extension at the original datasets** by setting the path to the `datasets/` folder in your clone, e.g.:

   ```bash
   export SYCOPHANCY_EVAL_DATASETS=/path/to/sycophancy-eval/datasets
   ```

   Or pass `--datasets-dir` to the generation script.

## Dataset format (unchanged)

Each generated JSONL line keeps the same structure as SycophancyEval:

- **`prompt`**: list of message dicts `{"type": "human", "content": "..."}` (tone is applied to the user message).
- **`base`**: original fields (e.g. `question`, `correct_answer`, `incorrect_answer` for `answer.jsonl`).
- **`metadata`**: includes `prompt_template` and **`tone`** (one of the five categories above).

## Generating tone-variant datasets

From this directory:

```bash
# Quick test with included sample (3 items)
python generate_tone_datasets.py --task answer --datasets-dir ./sample_data --out-dir ./datasets_by_tone

# Generate tone variants for full answer.jsonl (default: reads from ../sycophancy-eval/datasets)
python generate_tone_datasets.py --task answer --out-dir ./datasets_by_tone

# Use a specific path to the original datasets
python generate_tone_datasets.py --task answer --datasets-dir /path/to/sycophancy-eval/datasets --out-dir ./datasets_by_tone

# Generate for feedback.jsonl as well
python generate_tone_datasets.py --task feedback --datasets-dir /path/to/sycophancy-eval/datasets --out-dir ./datasets_by_tone

# Generate all supported tasks
python generate_tone_datasets.py --task all --datasets-dir /path/to/sycophancy-eval/datasets --out-dir ./datasets_by_tone
```

Output layout:

- `datasets_by_tone/answer/casual.jsonl`, `answer/unconfident_academic.jsonl`, …  
- `datasets_by_tone/feedback/casual.jsonl`, …

Each file is valid SycophancyEval-style JSONL; you can run evaluation with the original repo’s `utils.inference` (or your own) and then compare sycophancy rates **by tone**.

## Running evaluation (OpenAI and Claude)

Set the relevant API key(s), then run inference. Both OpenAI and Anthropic are supported; the script picks the client from the model name (`gpt` → OpenAI, `claude` → Anthropic).

**Single model:**
```bash
export OPENAI_API_KEY=sk-...
python run_inference.py --datasets-dir ./datasets_by_tone --task answer --model gpt-4o-mini --out results/results.jsonl

export ANTHROPIC_API_KEY=sk-ant-...
python run_inference.py --datasets-dir ./datasets_by_tone --task answer --model claude-3-5-sonnet-20241022 --out results/results.jsonl
```

**Both OpenAI and Claude in one go** (writes one JSONL per model):
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

python run_inference.py --datasets-dir ./datasets_by_tone --task answer \
  --models gpt-4o-mini,claude-3-5-sonnet-20241022 \
  --out results/results.jsonl
```
This creates `results/results_answer_gpt-4o-mini.jsonl` and `results/results_answer_claude-3-5-sonnet-20241022.jsonl`. Each row includes a `model` field for aggregation.

## Analysis

**`analysis/aggregate_by_tone.py`** reads one or more result JSONL files and aggregates **sycophancy rate and accuracy by tone** and by **(model, tone)** so you can compare both across tones and across OpenAI vs Claude.
```bash
# Single result file
python analysis/aggregate_by_tone.py --results results/results_answer_gpt-4o-mini.jsonl --out summary.json

# Combine OpenAI + Claude results  
python analysis/aggregate_by_tone.py \
  --results results/results_answer_gpt-4o-mini.jsonl results/results_answer_claude-3-5-sonnet-20241022.jsonl \
  --out summary_by_tone.json
```
Summary JSON includes `by_tone`, `by_tone_and_condition`, and `by_model_and_tone`. e.g. “Does high-stakes/urgent tone increase sycophancy relative to casual?”

## Running the experiment and figure

**One command** (generate datasets → run inference → aggregate → plot):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...   # optional, for Claude

python run_experiment.py --models gpt-4o-mini,claude-3-5-sonnet-20241022
```

Output: `results/summary_by_tone.json` and **`results/sycophancy_by_tone.html`** (open in a browser for the figure).

- **Quick test:** `python run_experiment.py --models gpt-4o-mini --limit 2`
- **Plot from existing results:** `python run_experiment.py --plot-only --results results/demo_results.jsonl`
- **Demo without API keys:** run `python scripts/make_demo_results.py` then the plot-only command above; open `results/sycophancy_by_tone.html`.

## Running without paid APIs

You can do the full experiment at no API cost in two ways.

### Option 1: Local model (Ollama) — real LLM, no payment

[Ollama](https://ollama.com) runs open-weight models on your machine and exposes an OpenAI-compatible API. No key or billing.

1. **Install Ollama** and pull a model:
   ```bash
   # Install from https://ollama.com, then:
   ollama pull llama3.2
   # or: ollama pull mistral
   ```

2. **Run the experiment** using the local server (no `OPENAI_API_KEY` needed):
   ```bash
   python run_experiment.py --models ollama/llama3.2 --base-url http://localhost:11434/v1
   ```
   Or with a smaller test: `python run_experiment.py --models ollama/llama3.2 --base-url http://localhost:11434/v1 --limit 2`

   Inference uses `http://localhost:11434/v1`; the script does not call OpenAI or Anthropic.

### Option 2: Synthetic demo data — no model, no install

Use pre-generated fake results to test the pipeline and produce a figure (e.g. for a methodology section):

```bash
python scripts/make_demo_results.py
python run_experiment.py --plot-only --results results/demo_results.jsonl
# Open results/sycophancy_by_tone.html
```

Label clearly in your report that these results are simulated.

### Option 3: Google Colab with GPU

To run the full experiment on a Colab **GPU** (faster than local CPU), use the included notebook:

1. Open **Google Colab**, then **File → Upload notebook** and choose `colab_run_experiment_gpu.ipynb` from this repo (or open the notebook from your cloned repo).
2. Set **Runtime → Change runtime type → GPU** (e.g. T4) and save.
3. Upload this repo to Colab: zip the `sycophancy-tone-extension` folder, upload the zip in the Colab Files panel (left sidebar), then in the notebook run: `!unzip -q sycophancy-tone-extension.zip -d /content` (and set `ROOT = Path("/content/sycophancy-tone-extension")` if needed).
4. Run the notebook cells **in order**. The first cells install Ollama, start the server, pull a model, and clone meg-tong/sycophancy-eval. **Run the “Verify Ollama is reachable” cell** before the experiment cell; if it fails, re-run the install and “Pull a model” cells (Ollama must be running in the same session). Then run the experiment cell. Download `results/sycophancy_by_tone.html` and `results/summary_by_tone.json` from the Files panel, or copy the `results/` folder to Google Drive.

**If every result shows `[Error: Connection error.]`:** The inference script could not reach Ollama. In Colab, run the cells in order in one session: (1) Install Ollama and start server, (2) Pull a model, (3) Clone repos, (4) **Verify Ollama is reachable** (this cell must succeed), (5) Run experiment. Do not restart the runtime between starting Ollama and running the experiment.

## Citation

If you use the original SycophancyEval data or code, cite:

```bibtex
@misc{sharma2023understanding,
  title={Towards Understanding Sycophancy in Language Models},
  author={Mrinank Sharma and Meg Tong and Tomasz Korbak and David Duvenaud and Amanda Askell and Samuel R. Bowman and Newton Cheng and Esin Durmus and Zac Hatfield-Dodds and Scott R. Johnston and Shauna Kravec and Timothy Maxwell and Sam McCandlish and Kamal Ndousse and Oliver Rausch and Nicholas Schiefer and Da Yan and Miranda Zhang and Ethan Perez},
  year={2023},
  eprint={2310.13548},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

This extension adds the tone dimension; cite the original paper and this repo/script set as appropriate.
