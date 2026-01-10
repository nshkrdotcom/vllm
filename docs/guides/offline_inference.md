# Offline Inference

Offline inference refers to batch processing of prompts without a running server. This is ideal for:

- Processing large datasets
- Batch evaluation
- One-time generation tasks
- Research and experimentation

## Basic Offline Inference

The `VLLM.llm/2` function creates an LLM instance for offline inference:

```elixir
VLLM.run(fn ->
  # Create LLM instance
  llm = VLLM.llm!("facebook/opt-125m")

  # Generate completions
  prompts = ["Hello, my name is", "The weather today is"]
  outputs = VLLM.generate!(llm, prompts)
end)
```

## LLM Configuration Options

```elixir
llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
  # Data type
  dtype: "auto",                    # "auto", "float16", "bfloat16", "float32"

  # Memory management
  gpu_memory_utilization: 0.9,      # Fraction of GPU memory to use
  max_model_len: 4096,              # Maximum sequence length

  # Parallelism
  tensor_parallel_size: 1,          # Number of GPUs for tensor parallelism

  # Quantization
  quantization: nil,                # "awq", "gptq", "squeezellm", etc.

  # Trust settings
  trust_remote_code: false          # Allow custom model code from HuggingFace
)
```

## Batch Processing

vLLM excels at batch processing with continuous batching:

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("facebook/opt-125m")
  params = VLLM.sampling_params!(temperature: 0.7, max_tokens: 100)

  # Large batch of prompts
  prompts = Enum.map(1..100, fn i ->
    "Story #{i}: Once upon a time,"
  end)

  # vLLM handles batching automatically
  start = System.monotonic_time(:millisecond)
  outputs = VLLM.generate!(llm, prompts, sampling_params: params)
  elapsed = System.monotonic_time(:millisecond) - start

  IO.puts("Processed #{length(prompts)} prompts in #{elapsed}ms")
  IO.puts("Throughput: #{Float.round(length(prompts) / (elapsed / 1000), 2)} prompts/sec")
end)
```

## Chat Mode for Offline Inference

Use chat format with instruction-tuned models:

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("Qwen/Qwen2-0.5B-Instruct")
  params = VLLM.sampling_params!(temperature: 0.7, max_tokens: 200)

  # Batch of conversations
  conversations = [
    [
      %{"role" => "user", "content" => "What is 2 + 2?"}
    ],
    [
      %{"role" => "user", "content" => "Name the planets in our solar system."}
    ],
    [
      %{"role" => "system", "content" => "You are a poet."},
      %{"role" => "user", "content" => "Write a haiku about coding."}
    ]
  ]

  outputs = VLLM.chat!(llm, conversations, sampling_params: params)
end)
```

## Memory-Efficient Processing

For large batches with limited GPU memory:

```elixir
VLLM.run(fn ->
  # Use lower memory utilization to leave room for KV cache
  llm = VLLM.llm!("facebook/opt-125m",
    gpu_memory_utilization: 0.7
  )

  # Process in chunks if needed
  all_prompts = Enum.to_list(1..1000) |> Enum.map(&"Prompt #{&1}:")

  chunk_size = 100
  all_prompts
  |> Enum.chunk_every(chunk_size)
  |> Enum.with_index(1)
  |> Enum.each(fn {chunk, idx} ->
    IO.puts("Processing chunk #{idx}...")
    outputs = VLLM.generate!(llm, chunk)
    # Process outputs...
  end)
end)
```

## Progress Tracking

vLLM shows progress by default via tqdm. Disable if needed:

```elixir
outputs = VLLM.generate!(llm, prompts,
  sampling_params: params,
  use_tqdm: false
)
```

## Tokenization

Access the tokenizer directly:

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("facebook/opt-125m")

  # Encode text to tokens
  token_ids = VLLM.encode!(llm, "Hello, world!")
  IO.inspect(token_ids, label: "Token IDs")
end)
```

## Performance Tips

1. **Maximize batch size**: vLLM is most efficient with larger batches
2. **Adjust `gpu_memory_utilization`**: Higher values allow more KV cache
3. **Use appropriate `max_model_len`**: Shorter = faster for short generations
4. **Consider quantization**: AWQ/GPTQ for memory-constrained scenarios
