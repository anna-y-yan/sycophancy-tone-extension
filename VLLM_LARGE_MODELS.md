# Running Large Models (Llama 3.3 70B, Qwen3 235B) with vLLM

Guide for running capable open-weight models with vLLM, following TA advice. vLLM is faster than Ollama and supports multi-GPU tensor parallelism.

---

## GPU requirements

| Model | VRAM (FP16) | Quantized (FP8/FP4) | Recommended GPUs |
|-------|-------------|---------------------|------------------|
| **Llama 3.3 70B** | ~140 GB | ~70 GB (FP8), ~35 GB (FP4) | 1× 80GB (A100/H100) for FP8; 2× 40GB (A100) for FP8; 1× B200 for FP4 |
| **Qwen3 235B** | ~470 GB | ~115 GB min (MoE) | 8× 80GB (A100/H100) recommended |

**Colab:** Free T4 (16GB) cannot run these. Colab Pro A100 (40GB) can run Llama 3.3 70B in FP8 with 2 GPUs if available, but typically you need **RunPod, Lambda Labs, Vast.ai, or a university cluster** for 70B and especially 235B.

---

## 1. Llama 3.3 70B (more accessible)

### Option A: FP8 on 1× 80GB GPU (A100/H100)

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
  --model nvidia/Llama-3.3-70B-Instruct-FP8 \
  --served-model-name llama3.3-70b \
  --tensor-parallel-size 1
```

### Option B: FP8 on 2× 40GB GPUs (e.g. 2× A100-40GB)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model nvidia/Llama-3.3-70B-Instruct-FP8 \
  --served-model-name llama3.3-70b \
  --tensor-parallel-size 2
```

### Option C: FP4 on 1× Blackwell (B200) – best efficiency

Uses NVIDIA’s pre-quantized FP4 checkpoint (smaller and faster on B200):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model nvidia/Llama-3.3-70B-Instruct-FP4 \
  --served-model-name llama3.3-70b \
  --tensor-parallel-size 1
```

**Note:** Accept the [Meta Llama 3 Community License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) before first use. No Hugging Face token needed for `nvidia/Llama-3.3-70B-Instruct-FP8` or `-FP4`.

---

## 2. Qwen3 235B (requires many GPUs)

Qwen3-235B-A22B is MoE (22B active params) but full weights need ~115 GB+ VRAM. Use **8× 80GB** GPUs:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-235B-A22B-Instruct \
  --served-model-name qwen3-235b \
  --tensor-parallel-size 8
```

With FP8 to reduce memory (if supported):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-235B-A22B-Instruct \
  --served-model-name qwen3-235b \
  --tensor-parallel-size 8 \
  --quantization fp8
```

You need a multi-GPU machine or cluster (e.g. 8× A100-80GB node on RunPod/Lambda).

---

## 3. Run the sycophancy experiment

Once the vLLM server is running (default `http://localhost:8000/v1`):

```bash
# Llama 3.3 70B
python run_experiment.py --datasets-dir /path/to/sycophancy-eval/datasets \
  --models llama3.3-70b \
  --base-url http://localhost:8000/v1 \
  --concurrency 32 \
  --regenerate

# Or run inference only
python run_inference.py --datasets-dir ./datasets_by_tone --task answer \
  --models llama3.3-70b \
  --base-url http://localhost:8000/v1 \
  --concurrency 32 \
  --out results/results.jsonl
```

**Model name:** Use the same name as `--served-model-name` (e.g. `llama3.3-70b`, `qwen3-235b`).

---

## 4. Where to get GPUs

| Provider | Typical options | Notes |
|----------|-----------------|-------|
| **RunPod** | 1–8× A100, pay per hour | Good for 70B (1–2 GPUs) and 235B (8 GPUs) |
| **Lambda Labs** | A100, H100 instances | Cloud or reserved |
| **Vast.ai** | Various GPUs, spot pricing | Cheaper, less predictable |
| **University cluster** | Depends on allocation | Request Slurm job with multiple GPUs |
| **Google Cloud / AWS** | A100, H100 VMs | More setup, pay-as-you-go |

**Checking your GPUs:**

```bash
nvidia-smi
```

Use `nvidia-smi` to see GPU count and VRAM per GPU, then set `--tensor-parallel-size` to the number of GPUs you want to use.

---

## 5. Quick reference: tensor parallelism

- `--tensor-parallel-size 1`: single GPU
- `--tensor-parallel-size 2`: 2 GPUs
- `--tensor-parallel-size 8`: 8 GPUs (e.g. for Qwen3 235B)

vLLM splits the model across GPUs automatically. Total VRAM must be enough for the model (see table above).
