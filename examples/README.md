# VLLM Examples

This directory contains comprehensive examples demonstrating VLLM capabilities for Elixir. VLLM wraps Python's vLLM library via SnakeBridge, providing high-throughput LLM inference.

## Prerequisites

**IMPORTANT: vLLM requires a CUDA-capable NVIDIA GPU.** If you don't have a compatible GPU, the inference examples will fail with CUDA errors.

```bash
# Install dependencies and set up Python environment
mix deps.get
mix snakebridge.setup

# Verify you have a CUDA-capable GPU
nvidia-smi
```

### GPU Requirements
- CUDA-capable NVIDIA GPU (e.g., RTX 3090, A100, V100)
- CUDA toolkit installed and configured
- Sufficient GPU memory for your chosen model (8GB+ recommended)

## Running Examples

Run any example individually:

```bash
mix run examples/basic.exs
```

Or run all examples with the test script:

```bash
./examples/run_all.sh
```

### Runtime options

Some examples accept CLI flags for overrides:

```bash
# Embeddings example (optional override)
mix run examples/embeddings.exs -- --model "BAAI/bge-large-en-v1.5"

# LoRA example (auto-downloads a public default adapter on first run)
mix run examples/lora.exs

# LoRA example (optional overrides)
mix run examples/lora.exs -- \
  --adapter /path/to/adapter \
  --model "your-base-model" \
  --name "adapter" \
  --prompt "Write a short SQL query to list users." \
  --rank 64

# Timeout example (optional overrides)
mix run examples/timeout_config.exs -- --model "facebook/opt-125m"
mix run examples/timeout_config.exs -- --prompt "Explain Elixir in one sentence."
```

The default LoRA adapter comes from `edbeeching/opt-125m-lora` (base model `facebook/opt-125m`)
and is downloaded automatically. This requires network access the first time it runs.

---

## Core Examples

### Basic Generation (`basic.exs`)

The foundational VLLM example showing core concepts:
- Creating an LLM instance
- Generating text completions
- Processing results

```elixir
{:ok, llm} = Vllm.LLM.new("facebook/opt-125m")
llm_ref = SnakeBridge.Ref.from_wire_format(llm)

runtime_opts =
  case llm_ref.pool_name do
    nil -> [session_id: llm_ref.session_id]
    pool_name -> [session_id: llm_ref.session_id, pool_name: pool_name]
  end

{:ok, outputs} =
  Vllm.LLM.generate(llm, ["Hello, my name is"], [], __runtime__: runtime_opts)
```

**Run:** `mix run examples/basic.exs`

---

### Sampling Parameters (`sampling_params.exs`)

Control text generation behavior:
- Temperature for randomness
- Top-p (nucleus) sampling
- Max tokens limit
- Stop sequences
- Multiple completions

```elixir
llm_ref = SnakeBridge.Ref.from_wire_format(llm)

runtime_opts =
  case llm_ref.pool_name do
    nil -> [session_id: llm_ref.session_id]
    pool_name -> [session_id: llm_ref.session_id, pool_name: pool_name]
  end

{:ok, params} =
  Vllm.SamplingParams.new([], temperature: 0.8, top_p: 0.95, max_tokens: 100, __runtime__: runtime_opts)

{:ok, outputs} =
  Vllm.LLM.generate(llm, [prompt], [], sampling_params: params, __runtime__: runtime_opts)
```

**Run:** `mix run examples/sampling_params.exs`

---

### Chat Completions (`chat.exs`)

Chat-style interactions with instruction-tuned models:
- System prompts
- Multi-turn conversations
- Batch chat processing

```elixir
llm_ref = SnakeBridge.Ref.from_wire_format(llm)

runtime_opts =
  case llm_ref.pool_name do
    nil -> [session_id: llm_ref.session_id]
    pool_name -> [session_id: llm_ref.session_id, pool_name: pool_name]
  end

messages = [[
  %{"role" => "system", "content" => "You are helpful."},
  %{"role" => "user", "content" => "Hello!"}
]]

{:ok, outputs} = Vllm.LLM.chat(llm, messages, [], __runtime__: runtime_opts)
```

**Run:** `mix run examples/chat.exs`

---

### Batch Inference (`batch_inference.exs`)

High-throughput batch processing:
- Processing multiple prompts efficiently
- Continuous batching
- Performance measurement

```elixir
llm_ref = SnakeBridge.Ref.from_wire_format(llm)

runtime_opts =
  case llm_ref.pool_name do
    nil -> [session_id: llm_ref.session_id]
    pool_name -> [session_id: llm_ref.session_id, pool_name: pool_name]
  end

prompts = ["Prompt 1", "Prompt 2", "Prompt 3", ...]
{:ok, outputs} = Vllm.LLM.generate(llm, prompts, [], sampling_params: params, __runtime__: runtime_opts)
```

**Run:** `mix run examples/batch_inference.exs`

---

## Advanced Examples

### Structured Output (`structured_output.exs`)

Guided generation for structured outputs:
- JSON schema constraints
- Regex patterns
- Choice constraints

```elixir
{:ok, params} = Vllm.SamplingParams.new([], structured_outputs: %{choice: ["yes", "no", "maybe"]})
```

**Run:** `mix run examples/structured_output.exs`

---

### Quantization (`quantization.exs`)

Memory-efficient inference with quantized models:
- AWQ quantization
- GPTQ quantization
- Memory comparison

```elixir
{:ok, llm} = Vllm.LLM.new("TheBloke/Llama-2-7B-AWQ", quantization: "awq")
```

**Run:** `mix run examples/quantization.exs`

---

### Multi-GPU (`multi_gpu.exs`)

Distributed inference across GPUs:
- Tensor parallelism
- Pipeline parallelism
- Memory utilization

```elixir
{:ok, llm} = Vllm.LLM.new("meta-llama/Llama-2-13b-hf",
  tensor_parallel_size: 2,
  gpu_memory_utilization: 0.9
)
```

**Run:** `mix run examples/multi_gpu.exs`

---

### Embeddings (`embeddings.exs`)

Vector embeddings for semantic search:
- Loading embedding models
- Batch embedding
- Use cases

```elixir
{:ok, llm} = Vllm.LLM.new("intfloat/e5-mistral-7b-instruct", runner: "pooling")
llm_ref = SnakeBridge.Ref.from_wire_format(llm)

runtime_opts =
  case llm_ref.pool_name do
    nil -> [session_id: llm_ref.session_id]
    pool_name -> [session_id: llm_ref.session_id, pool_name: pool_name]
  end

{:ok, outputs} = Vllm.LLM.embed(llm, ["Hello, world!"], __runtime__: runtime_opts)
```

**Run:** `mix run examples/embeddings.exs`

---

### LoRA Adapters (`lora.exs`)

Fine-tuned model serving:
- Loading LoRA adapters
- Multi-LoRA serving
- Configuration

```elixir
{:ok, llm} = Vllm.LLM.new("meta-llama/Llama-2-7b-hf", enable_lora: true)
llm_ref = SnakeBridge.Ref.from_wire_format(llm)

runtime_opts =
  case llm_ref.pool_name do
    nil -> [session_id: llm_ref.session_id]
    pool_name -> [session_id: llm_ref.session_id, pool_name: pool_name]
  end

{:ok, lora} =
  Vllm.BeamSearch.LoRARequest.new(["my-adapter", 1, "/path/to/adapter"], __runtime__: runtime_opts)

{:ok, outputs} = Vllm.LLM.generate(llm, [prompt], [], lora_request: lora, __runtime__: runtime_opts)
```

**Run:** `mix run examples/lora.exs`

---

### Timeout Configuration (`timeout_config.exs`)

Configure timeouts for long-running operations:
- Timeout profiles
- Per-call overrides
- Helper functions

```elixir
llm_ref = SnakeBridge.Ref.from_wire_format(llm)

runtime_opts =
  case llm_ref.pool_name do
    nil -> [session_id: llm_ref.session_id]
    pool_name -> [session_id: llm_ref.session_id, pool_name: pool_name]
  end

{:ok, outputs} =
  Vllm.LLM.generate(llm, prompts, [],
    sampling_params: params,
    __runtime__: Keyword.merge(runtime_opts, timeout_profile: :batch_job)
  )
```

**Run:** `mix run examples/timeout_config.exs`

---

### Wrapper API (`direct_api.exs`)

Demonstrates wrapper-only usage:

1. **Generated wrappers** (type-safe): `Vllm.LLM.new/2`, `Vllm.SamplingParams.new/2`
2. **Runtime attribute access** for Python refs via `SnakeBridge.Runtime.get_attr/2`

```elixir
{:ok, llm} = Vllm.LLM.new("facebook/opt-125m")
llm_ref = SnakeBridge.Ref.from_wire_format(llm)

runtime_opts =
  case llm_ref.pool_name do
    nil -> [session_id: llm_ref.session_id]
    pool_name -> [session_id: llm_ref.session_id, pool_name: pool_name]
  end

{:ok, params} = Vllm.SamplingParams.new([], temperature: 0.8, __runtime__: runtime_opts)
{:ok, outputs} = Vllm.LLM.generate(llm, ["Hello"], [], sampling_params: params, __runtime__: runtime_opts)
```

**Run:** `mix run examples/direct_api.exs`

---

## Running All Examples

The `run_all.sh` script runs all examples sequentially with:
- Colorized output
- Per-example timing
- Pass/fail summary
- Automatic timeout handling

```bash
# Run with default timeout
./examples/run_all.sh

# Run with custom timeout (300s per example)
VLLM_RUN_TIMEOUT_SECONDS=300 ./examples/run_all.sh

# Disable timeout
VLLM_RUN_TIMEOUT_SECONDS=0 ./examples/run_all.sh
```

## Example Index

| Example | Focus | Description |
|---------|-------|-------------|
| `basic.exs` | Core | Simple text generation |
| `sampling_params.exs` | Core | Generation control parameters |
| `chat.exs` | Core | Chat completions |
| `batch_inference.exs` | Performance | High-throughput batching |
| `structured_output.exs` | Advanced | Constrained generation |
| `quantization.exs` | Advanced | Memory-efficient models |
| `multi_gpu.exs` | Advanced | Distributed inference |
| `embeddings.exs` | Advanced | Vector embeddings |
| `lora.exs` | Advanced | Fine-tuned adapters |
| `timeout_config.exs` | Configuration | Timeout settings |
| `direct_api.exs` | Advanced | Wrapper-only API usage |

## Troubleshooting

### No CUDA-Capable GPU / CUDA Errors
```
CUDA error: no kernel image is available for execution on the device
```
or
```
RuntimeError: CUDA error
```
- **vLLM requires a CUDA-capable NVIDIA GPU** - it cannot run on CPU-only systems
- Verify your GPU is detected: `nvidia-smi`
- Ensure CUDA toolkit is properly installed
- Check GPU compute capability matches vLLM requirements (compute capability 7.0+)

### CUDA Out of Memory
```
CUDA out of memory
```
- Reduce `gpu_memory_utilization`
- Use smaller model
- Use quantized model

### Model Not Found
```
Model not found
```
- Check model name on HuggingFace
- Check internet connection

### Timeout Errors
For long operations, increase timeout:
```elixir
Vllm.LLM.generate(llm, prompts, [],
  __runtime__: Keyword.merge(runtime_opts, timeout_profile: :batch_job)
)
```

### Python/vLLM Not Installed
```
Module vllm not found
```
Run: `mix snakebridge.setup`
