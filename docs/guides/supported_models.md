# Supported Models

vLLM supports a wide variety of open-source models from HuggingFace. This guide lists major supported model families.

## Transformer LLMs

### Llama Family

```elixir
# Llama 2
VLLM.llm!("meta-llama/Llama-2-7b-hf")
VLLM.llm!("meta-llama/Llama-2-13b-hf")
VLLM.llm!("meta-llama/Llama-2-70b-hf")
VLLM.llm!("meta-llama/Llama-2-7b-chat-hf")

# Llama 3
VLLM.llm!("meta-llama/Meta-Llama-3-8B")
VLLM.llm!("meta-llama/Meta-Llama-3-70B")
VLLM.llm!("meta-llama/Meta-Llama-3-8B-Instruct")

# Llama 3.1
VLLM.llm!("meta-llama/Llama-3.1-8B")
VLLM.llm!("meta-llama/Llama-3.1-70B")
VLLM.llm!("meta-llama/Llama-3.1-405B")

# Llama 3.2
VLLM.llm!("meta-llama/Llama-3.2-1B")
VLLM.llm!("meta-llama/Llama-3.2-3B")
VLLM.llm!("meta-llama/Llama-3.2-1B-Instruct")
```

Note: Some Llama models are gated. Accept the license and request access on Hugging Face.

### Mistral Family

```elixir
VLLM.llm!("mistralai/Mistral-7B-v0.1")
VLLM.llm!("mistralai/Mistral-7B-Instruct-v0.2")
VLLM.llm!("mistralai/Mistral-Nemo-Base-2407")
```

### Qwen Family

```elixir
VLLM.llm!("Qwen/Qwen2-7B")
VLLM.llm!("Qwen/Qwen2-72B")
VLLM.llm!("Qwen/Qwen2.5-7B-Instruct")
VLLM.llm!("Qwen/Qwen2.5-72B-Instruct")
```

### Gemma Family

```elixir
VLLM.llm!("google/gemma-2b")
VLLM.llm!("google/gemma-7b")
VLLM.llm!("google/gemma-2-9b")
VLLM.llm!("google/gemma-2-27b")
```

### Phi Family

```elixir
VLLM.llm!("microsoft/phi-2")
VLLM.llm!("microsoft/Phi-3-mini-4k-instruct")
VLLM.llm!("microsoft/Phi-3.5-mini-instruct")
```

## Mixture-of-Experts (MoE)

```elixir
# Mixtral
VLLM.llm!("mistralai/Mixtral-8x7B-v0.1")
VLLM.llm!("mistralai/Mixtral-8x7B-Instruct-v0.1")
VLLM.llm!("mistralai/Mixtral-8x22B-v0.1")

# DeepSeek
VLLM.llm!("deepseek-ai/deepseek-moe-16b-base")
VLLM.llm!("deepseek-ai/DeepSeek-V2")
VLLM.llm!("deepseek-ai/DeepSeek-V3")

# Qwen MoE
VLLM.llm!("Qwen/Qwen1.5-MoE-A2.7B")
```

## Code Models

```elixir
# CodeLlama
VLLM.llm!("codellama/CodeLlama-7b-hf")
VLLM.llm!("codellama/CodeLlama-34b-Instruct-hf")

# StarCoder
VLLM.llm!("bigcode/starcoder2-15b")

# DeepSeek Coder
VLLM.llm!("deepseek-ai/deepseek-coder-6.7b-base")
```

## Embedding Models

```elixir
# E5 Mistral
VLLM.llm!("intfloat/e5-mistral-7b-instruct", task: "embed")

# BGE
VLLM.llm!("BAAI/bge-large-en-v1.5", task: "embed")
```

## Multimodal Models

```elixir
# LLaVA
VLLM.llm!("llava-hf/llava-1.5-7b-hf")
VLLM.llm!("llava-hf/llava-v1.6-mistral-7b-hf")

# Qwen-VL
VLLM.llm!("Qwen/Qwen2-VL-7B-Instruct")
```

## Quantized Models

Many models are available pre-quantized:

```elixir
# AWQ Quantized
VLLM.llm!("TheBloke/Llama-2-7B-AWQ", quantization: "awq")
VLLM.llm!("TheBloke/Llama-2-13B-AWQ", quantization: "awq")

# GPTQ Quantized
VLLM.llm!("TheBloke/Llama-2-7B-GPTQ", quantization: "gptq")
```

## Small Models (for Testing)

```elixir
# OPT (small, fast)
VLLM.llm!("facebook/opt-125m")
VLLM.llm!("facebook/opt-350m")
VLLM.llm!("facebook/opt-1.3b")

# GPT-2
VLLM.llm!("gpt2")
VLLM.llm!("gpt2-medium")
```

## Model Selection Tips

1. **For chat/instruction following**: Use "-Instruct" or "-chat" variants
2. **For memory constraints**: Use quantized versions (AWQ, GPTQ)
3. **For multi-GPU**: Use larger models with tensor parallelism
4. **For testing**: Use small models like OPT-125m

## Gated Models

Some models require accepting terms on HuggingFace:

1. Visit the model page on HuggingFace
2. Accept the license agreement
3. Set `HF_TOKEN` environment variable

```bash
export HF_TOKEN="your-huggingface-token"
```

## Adding New Models

vLLM supports most transformer-based architectures. If a model isn't working:

1. Check if it's on the [vLLM supported models list](https://docs.vllm.ai/en/latest/models/supported_models.html)
2. Try with `trust_remote_code: true` for custom architectures
3. Check vLLM GitHub issues for known compatibility issues
