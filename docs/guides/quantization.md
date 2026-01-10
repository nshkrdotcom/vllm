# Quantization

Quantization reduces model memory footprint by using lower-precision number formats. vLLM supports multiple quantization methods.

## Why Quantization?

| Precision | Memory per Parameter | Relative Size |
|-----------|---------------------|---------------|
| FP32 | 4 bytes | 100% |
| FP16/BF16 | 2 bytes | 50% |
| INT8 | 1 byte | 25% |
| INT4 | 0.5 bytes | 12.5% |

A 7B parameter model:
- FP32: ~28 GB
- FP16: ~14 GB
- INT8: ~7 GB
- INT4: ~3.5 GB

## Supported Methods

### AWQ (Activation-aware Weight Quantization)

High-quality 4-bit quantization with excellent performance:

```elixir
llm = VLLM.llm!("TheBloke/Llama-2-7B-AWQ",
  quantization: "awq"
)
```

### GPTQ

Popular post-training quantization method:

```elixir
llm = VLLM.llm!("TheBloke/Llama-2-7B-GPTQ",
  quantization: "gptq"
)
```

### SqueezeLLM

Sensitivity-based non-uniform quantization:

```elixir
llm = VLLM.llm!("squeezellm/Llama-2-7b-squeezellm",
  quantization: "squeezellm"
)
```

### FP8

8-bit floating point quantization:

```elixir
llm = VLLM.llm!("neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
  quantization: "fp8"
)
```

### BitsAndBytes

4-bit quantization via the bitsandbytes library:

```elixir
llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
  quantization: "bitsandbytes",
  load_format: "bitsandbytes"
)
```

## Using Pre-Quantized Models

The easiest approach is using pre-quantized models from HuggingFace:

```elixir
# AWQ models from TheBloke
VLLM.llm!("TheBloke/Llama-2-7B-AWQ", quantization: "awq")
VLLM.llm!("TheBloke/Llama-2-13B-AWQ", quantization: "awq")
VLLM.llm!("TheBloke/Mistral-7B-Instruct-v0.2-AWQ", quantization: "awq")

# GPTQ models from TheBloke
VLLM.llm!("TheBloke/Llama-2-7B-GPTQ", quantization: "gptq")
VLLM.llm!("TheBloke/CodeLlama-34B-GPTQ", quantization: "gptq")
```

## Quantized KV Cache

In addition to model quantization, you can quantize the KV cache:

```elixir
llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
  kv_cache_dtype: "fp8"  # Reduce KV cache memory by 50%
)
```

## Performance Comparison

| Method | Memory Reduction | Speed Impact | Quality Impact |
|--------|-----------------|--------------|----------------|
| AWQ | ~75% | +20-50% | Minimal |
| GPTQ | ~75% | +10-30% | Minimal |
| FP8 | ~50% | +10-20% | Negligible |
| INT8 | ~50% | +10-30% | Negligible |

## When to Use Quantization

**Use quantization when:**
- GPU memory is limited
- Running larger models
- Need to maximize throughput
- Quality loss is acceptable

**Avoid quantization when:**
- Maximum quality is required
- Running small models
- Memory is not constrained

## Example: Running Large Models

Run a 70B model on consumer hardware:

```elixir
# 70B model quantized to fit on 2x 24GB GPUs
llm = VLLM.llm!("TheBloke/Llama-2-70B-AWQ",
  quantization: "awq",
  tensor_parallel_size: 2,
  gpu_memory_utilization: 0.95
)
```

## Quality Considerations

Quantization can affect output quality, especially for:
- Mathematical reasoning
- Code generation
- Precise factual recall

For critical applications, benchmark quantized vs. full-precision models on your specific use case.
