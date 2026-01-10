# Quickstart Guide

This guide will help you get started with VLLM for Elixir, providing high-throughput LLM inference via vLLM.

## Prerequisites

- Elixir 1.18 or later
- Python 3.8 or later
- CUDA-capable GPU (recommended) or CPU-only mode

## Installation

Add VLLM to your `mix.exs` dependencies:

```elixir
def deps do
  [
    {:vllm, "~> 0.1.0"}
  ]
end
```

Fetch dependencies and set up the Python environment:

```bash
mix deps.get
mix snakebridge.setup
```

This will install vLLM and its dependencies in a managed Python environment.

## Your First Generation

Here's a minimal example to generate text:

```elixir
VLLM.run(fn ->
  # Load a small model for testing
  llm = VLLM.llm!("facebook/opt-125m")

  # Generate completions
  outputs = VLLM.generate!(llm, "Hello, my name is")

  # Print the result
  output = Enum.at(outputs, 0)
  completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
  text = VLLM.attr!(completion, "text")
  IO.puts(text)
end)
```

Save this as `hello_vllm.exs` and run:

```bash
mix run hello_vllm.exs
```

## Understanding the Output

vLLM returns `RequestOutput` objects with the following structure:

- `prompt` - The original input prompt
- `outputs` - List of `CompletionOutput` objects
  - `text` - The generated text
  - `token_ids` - List of generated token IDs
  - `finish_reason` - Why generation stopped ("length", "stop", etc.)

## Controlling Generation

Use `SamplingParams` to control text generation:

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("facebook/opt-125m")

  # Create sampling parameters
  params = VLLM.sampling_params!(
    temperature: 0.8,    # Higher = more random
    top_p: 0.95,         # Nucleus sampling
    max_tokens: 100,     # Maximum tokens to generate
    stop: ["\\n"]        # Stop at newline
  )

  outputs = VLLM.generate!(llm, "The secret to happiness is",
    sampling_params: params
  )
end)
```

## Chat Mode

For instruction-tuned models, use the chat interface:

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("Qwen/Qwen2-0.5B-Instruct")

  messages = [[
    %{"role" => "system", "content" => "You are a helpful assistant."},
    %{"role" => "user", "content" => "Explain quantum computing in simple terms."}
  ]]

  params = VLLM.sampling_params!(temperature: 0.7, max_tokens: 200)
  outputs = VLLM.chat!(llm, messages, sampling_params: params)
end)
```

## Batch Processing

Process multiple prompts efficiently:

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("facebook/opt-125m")
  params = VLLM.sampling_params!(temperature: 0.7, max_tokens: 50)

  prompts = [
    "The capital of France is",
    "Machine learning is",
    "The best way to learn programming is"
  ]

  # vLLM processes these efficiently with continuous batching
  outputs = VLLM.generate!(llm, prompts, sampling_params: params)

  Enum.each(outputs, fn output ->
    prompt = VLLM.attr!(output, "prompt")
    completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
    IO.puts("#{prompt}#{VLLM.attr!(completion, "text")}")
  end)
end)
```

## Next Steps

- [Sampling Parameters](sampling_params.md) - Fine-tune generation behavior
- [Configuration](configuration.md) - Model and engine options
- [Supported Models](supported_models.md) - Full list of supported models
- [Examples](../examples/README.md) - Comprehensive code examples
