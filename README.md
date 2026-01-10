<p align="center">
  <img src="assets/vllm.svg" alt="VLLM" width="200">
</p>

<h1 align="center">VLLM</h1>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone - in Elixir
</h3>

<p align="center">
  <a href="https://hex.pm/packages/vllm"><img src="https://img.shields.io/hexpm/v/vllm.svg" alt="Hex.pm"></a>
  <a href="https://hexdocs.pm/vllm"><img src="https://img.shields.io/badge/hex-docs-blue.svg" alt="Documentation"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<p align="center">
| <a href="https://hexdocs.pm/vllm"><b>Documentation</b></a> | <a href="https://github.com/nshkrdotcom/vllm"><b>GitHub</b></a> | <a href="https://docs.vllm.ai"><b>vLLM Python Docs</b></a> |
</p>

---

## About

VLLM is an Elixir client for [vLLM](https://github.com/vllm-project/vllm), the high-throughput LLM inference engine. It provides transparent access to vLLM's powerful features through [SnakeBridge](https://github.com/nshkrdotcom/snakebridge).

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

**vLLM is fast with:**

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

**vLLM is flexible and easy to use with:**

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, Arm CPUs, and TPU
- Prefix caching support
- Multi-LoRA support

**vLLM seamlessly supports most popular open-source models on HuggingFace, including:**

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

## Requirements

**IMPORTANT: vLLM requires a CUDA-capable NVIDIA GPU.** The library cannot run on CPU-only systems.

- NVIDIA GPU with CUDA support (compute capability 7.0+)
- CUDA toolkit installed
- 8GB+ GPU memory recommended (varies by model)

Verify your GPU setup:
```bash
nvidia-smi
```

## Installation

Add `vllm` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:vllm, "~> 0.1.1"}
  ]
end
```

Then fetch dependencies and set up Python:

```bash
mix deps.get
mix snakebridge.setup
```

## Quick Start

### Basic Text Generation

```elixir
VLLM.run(fn ->
  # Create an LLM instance
  llm = VLLM.llm!("facebook/opt-125m")

  # Generate text
  outputs = VLLM.generate!(llm, ["Hello, my name is"])

  # Process results
  Enum.each(outputs, fn output ->
    prompt = VLLM.attr!(output, "prompt")
    generated = VLLM.attr!(output, "outputs") |> Enum.at(0)
    text = VLLM.attr!(generated, "text")
    IO.puts("#{prompt}#{text}")
  end)
end)
```

### Chat Completions

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("Qwen/Qwen2-0.5B-Instruct")

  messages = [[
    %{"role" => "system", "content" => "You are a helpful assistant."},
    %{"role" => "user", "content" => "What is the capital of France?"}
  ]]

  outputs = VLLM.chat!(llm, messages)

  output = Enum.at(outputs, 0)
  completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
  IO.puts(VLLM.attr!(completion, "text"))
end)
```

### Sampling Parameters

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("facebook/opt-125m")

  params = VLLM.sampling_params!(
    temperature: 0.8,
    top_p: 0.95,
    max_tokens: 100
  )

  outputs = VLLM.generate!(llm, ["Once upon a time"], sampling_params: params)
end)
```

### Batch Processing

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("facebook/opt-125m")
  params = VLLM.sampling_params!(temperature: 0.7, max_tokens: 50)

  prompts = [
    "The meaning of life is",
    "Artificial intelligence will",
    "The best programming language is"
  ]

  # Process all prompts efficiently with continuous batching
  outputs = VLLM.generate!(llm, prompts, sampling_params: params)
end)
```

## Features

### Quantization

Load quantized models for memory-efficient inference:

```elixir
llm = VLLM.llm!("TheBloke/Llama-2-7B-AWQ", quantization: "awq")
llm = VLLM.llm!("TheBloke/Llama-2-7B-GPTQ", quantization: "gptq")
```

### Multi-GPU / Tensor Parallelism

Distribute large models across multiple GPUs:

```elixir
llm = VLLM.llm!("meta-llama/Llama-2-13b-hf",
  tensor_parallel_size: 2,
  gpu_memory_utilization: 0.9
)
```

### LoRA Adapters

Serve fine-tuned models with LoRA:

```elixir
llm = VLLM.llm!("meta-llama/Llama-2-7b-hf", enable_lora: true)
lora = VLLM.lora_request!("my-adapter", 1, "/path/to/adapter")
outputs = VLLM.generate!(llm, prompt, lora_request: lora)
```

### Structured Outputs

Constrain generation with JSON schema, regex, or choices:

```elixir
# JSON schema
guided = VLLM.guided_decoding_params!(
  json: ~s({"type": "object", "properties": {"name": {"type": "string"}}})
)

# Regex pattern
guided = VLLM.guided_decoding_params!(regex: "[0-9]{3}-[0-9]{4}")

# Choice
guided = VLLM.guided_decoding_params!(choice: ["yes", "no", "maybe"])
```

### Embeddings

Generate vector embeddings with pooling models:

```elixir
llm = VLLM.llm!("intfloat/e5-mistral-7b-instruct", runner: "pooling")
embeddings = VLLM.embed!(llm, ["Hello, world!", "How are you?"])
```

## Timeout Configuration

VLLM uses SnakeBridge's timeout architecture optimized for ML workloads:

| Profile | Timeout | Use Case |
|---------|---------|----------|
| `:default` | 2 min | Standard Python calls |
| `:streaming` | 30 min | Streaming responses |
| `:ml_inference` | 10 min | LLM inference (default) |
| `:batch_job` | 1 hour | Long-running batches |

Override per-call:

```elixir
VLLM.generate!(llm, prompts,
  sampling_params: params,
  __runtime__: [timeout_profile: :batch_job]
)
```

## Architecture

```
Elixir (VLLM)
    │
    ▼
SnakeBridge.call/4
    │
    ▼
Snakepit gRPC
    │
    ▼
Python vLLM
    │
    ▼
GPU/TPU Inference
```

## Documentation

- [API Reference](https://hexdocs.pm/vllm)
- [Quickstart Guide](https://hexdocs.pm/vllm/quickstart.html)
- [Examples](https://hexdocs.pm/vllm/examples.html)
- [Configuration Guide](https://hexdocs.pm/vllm/configuration.html)
- [vLLM Python Documentation](https://docs.vllm.ai)

## Examples

See the [examples](https://hexdocs.pm/vllm/examples.html) for comprehensive usage:

- `basic.exs` - Simple text generation
- `sampling_params.exs` - Generation control
- `chat.exs` - Chat completions
- `batch_inference.exs` - High-throughput batching
- `structured_output.exs` - Constrained generation
- `quantization.exs` - Memory-efficient models
- `multi_gpu.exs` - Distributed inference
- `embeddings.exs` - Vector embeddings
- `lora.exs` - Fine-tuned adapters
- `timeout_config.exs` - Timeout settings
- `direct_api.exs` - Raw Python access

Run all examples:

```bash
./examples/run_all.sh
```

## Citation

If you use vLLM for your research, please cite the [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/nshkrdotcom/vllm)
- [Hex Package](https://hex.pm/packages/vllm)
- [Documentation](https://hexdocs.pm/vllm)
- [vLLM Python Project](https://github.com/vllm-project/vllm)
- [SnakeBridge](https://github.com/nshkrdotcom/snakebridge)
