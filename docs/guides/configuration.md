# Configuration Guide

This guide covers all configuration options for VLLM.

## Model Configuration

### Basic Options

```elixir
llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
  # Data type for model weights
  dtype: "auto",              # "auto", "float16", "bfloat16", "float32"

  # Maximum sequence length
  max_model_len: 4096,

  # Trust remote code from HuggingFace
  trust_remote_code: false
)
```

### Memory Configuration

```elixir
llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
  # GPU memory utilization (0.0 to 1.0)
  gpu_memory_utilization: 0.9,

  # Swap space for CPU offloading (GB)
  swap_space: 4,

  # CPU offload GB
  cpu_offload_gb: 0
)
```

### Parallelism Configuration

```elixir
llm = VLLM.llm!("meta-llama/Llama-2-70b-hf",
  # Tensor parallelism (split layers across GPUs)
  tensor_parallel_size: 4,

  # Pipeline parallelism (split stages across GPUs)
  pipeline_parallel_size: 2,

  # Distributed executor backend
  distributed_executor_backend: "ray"  # or "mp"
)
```

### Quantization Configuration

```elixir
llm = VLLM.llm!("TheBloke/Llama-2-7B-AWQ",
  # Quantization method
  quantization: "awq",        # "awq", "gptq", "squeezellm", "fp8", etc.

  # Quantized KV cache
  kv_cache_dtype: "auto"      # "auto", "fp8"
)
```

## Snakebridge Configuration

Configure in `config/config.exs`:

```elixir
import Config

config :snakebridge,
  verbose: false,
  runtime: [
    # Set vLLM to use ML inference timeout profile
    library_profiles: %{"vllm" => :ml_inference}
  ]

# Configure snakepit at compile time so SnakeBridge installs Python deps
# into the same venv used at runtime (ConfigHelper is not available here).
project_root = Path.expand("..", __DIR__)
snakebridge_venv =
  [
    System.get_env("SNAKEBRIDGE_VENV"),
    Path.join(project_root, ".venv"),
    Path.expand("../snakebridge/.venv", __DIR__)
  ]
  |> Enum.find(fn path -> is_binary(path) and File.dir?(path) end)

python_executable =
  if snakebridge_venv do
    [
      Path.join([snakebridge_venv, "bin", "python3"]),
      Path.join([snakebridge_venv, "bin", "python"]),
      Path.join([snakebridge_venv, "Scripts", "python.exe"]),
      Path.join([snakebridge_venv, "Scripts", "python"])
    ]
    |> Enum.find(&File.exists?/1)
  end

if snakebridge_venv do
  config :snakebridge, venv_path: snakebridge_venv
end

if python_executable do
  config :snakepit, python_executable: python_executable
end

config :snakepit, environment: config_env()

config :logger, level: :warning
```

### Compile-Time Bindings (SnakeBridge)

VLLM uses SnakeBridge to generate Elixir wrappers for vLLM’s documented public API. In `mix.exs`,
`python_deps` is configured with `module_mode: :docs` using a committed docs manifest
(`priv/snakebridge/vllm.docs.json`). This keeps generation deterministic and avoids walking vLLM’s
deep internal module tree at build time.

Additionally, `max_class_methods` is enabled to prevent inheritance-heavy internal classes from
producing extremely large generated wrapper modules.

Configure in `config/runtime.exs`:

```elixir
import Config

# Auto-configure Snakepit (and apply vLLM safety defaults)
VLLM.ConfigHelper.configure_snakepit!()
```

### vLLM v1 Multiprocessing (vLLM 0.14+)

vLLM can spawn subprocesses for its engine core. Under Snakepit, this must be
paired with reliable process-group cleanup to avoid orphan GPU processes.

Configure via:

```elixir
# config/config.exs
config :vllm, v1_multiprocessing: :auto  # :auto | :on | :off
```

Semantics:

- `:off` - forces single-process mode (`VLLM_ENABLE_V1_MULTIPROCESSING=0`)
- `:on` - forces multiprocessing and fails fast unless Snakepit process-group
  cleanup is available
- `:auto` - enables multiprocessing only when safe; otherwise forces off with a warning

## Timeout Configuration

### Global Timeouts

```elixir
# config/config.exs
config :snakebridge,
  runtime: [
    library_profiles: %{"vllm" => :batch_job}  # 1 hour timeout
  ]
```

### Per-Call Timeouts

```elixir
# Use timeout profile
outputs = VLLM.generate!(llm, prompts,
  __runtime__: [timeout_profile: :batch_job]
)

# Use exact milliseconds
outputs = VLLM.generate!(llm, prompts,
  __runtime__: [timeout: 300_000]  # 5 minutes
)

# Using helpers
opts = VLLM.with_timeout([sampling_params: params], timeout_profile: :ml_inference)
outputs = VLLM.generate!(llm, prompts, opts)
```

### Timeout Profiles

| Profile | Duration | Use Case |
|---------|----------|----------|
| `:default` | 2 min | Standard calls |
| `:streaming` | 30 min | Streaming responses |
| `:ml_inference` | 10 min | LLM inference (recommended) |
| `:batch_job` | 1 hour | Large batch processing |

## Environment Variables

vLLM respects these environment variables:

```bash
# HuggingFace token for gated models
export HF_TOKEN="your-token"

# Use ModelScope instead of HuggingFace
export VLLM_USE_MODELSCOPE=1

# Specify CUDA devices
export CUDA_VISIBLE_DEVICES="0,1"

# Disable warnings
export VLLM_LOGGING_LEVEL=ERROR
```

## Model Loading

### From HuggingFace Hub

```elixir
# Public model
llm = VLLM.llm!("facebook/opt-125m")

# Gated model (requires HF_TOKEN)
llm = VLLM.llm!("meta-llama/Llama-2-7b-hf")
```

### From Local Path

```elixir
llm = VLLM.llm!("/path/to/local/model")
```

### Specific Revision

```elixir
llm = VLLM.llm!("facebook/opt-125m",
  revision: "main"  # or specific commit hash
)
```

## LoRA Configuration

```elixir
llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
  enable_lora: true,
  max_lora_rank: 64,
  max_loras: 4,
  lora_extra_vocab_size: 256
)
```

## Structured Output Configuration

```elixir
llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
  # Enable guided decoding
  guided_decoding_backend: "outlines"  # or "lm-format-enforcer"
)
```

## Performance Tuning

### For Maximum Throughput

```elixir
llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
  gpu_memory_utilization: 0.95,
  max_num_batched_tokens: 8192,
  max_num_seqs: 256
)
```

### For Minimum Latency

```elixir
llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
  gpu_memory_utilization: 0.8,
  max_num_seqs: 1
)
```

### For Memory-Constrained Systems

```elixir
llm = VLLM.llm!("TheBloke/Llama-2-7B-AWQ",
  quantization: "awq",
  gpu_memory_utilization: 0.7,
  max_model_len: 2048
)
```
