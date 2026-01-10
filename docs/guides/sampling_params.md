# Sampling Parameters

`SamplingParams` controls how vLLM generates text. Understanding these parameters is essential for getting the output quality and style you need.

## Creating Sampling Parameters

```elixir
params = VLLM.sampling_params!(
  temperature: 0.8,
  top_p: 0.95,
  max_tokens: 100
)

outputs = VLLM.generate!(llm, prompt, sampling_params: params)
```

## Temperature

Controls randomness in generation. Higher values produce more diverse outputs.

```elixir
# Deterministic (greedy decoding)
params = VLLM.sampling_params!(temperature: 0.0, max_tokens: 50)

# Low temperature (focused, consistent)
params = VLLM.sampling_params!(temperature: 0.3, max_tokens: 50)

# Medium temperature (balanced)
params = VLLM.sampling_params!(temperature: 0.7, max_tokens: 50)

# High temperature (creative, diverse)
params = VLLM.sampling_params!(temperature: 1.2, max_tokens: 50)
```

## Top-p (Nucleus Sampling)

Limits sampling to tokens comprising the top `p` probability mass.

```elixir
# Only consider tokens in top 90% probability mass
params = VLLM.sampling_params!(top_p: 0.9, temperature: 0.7)

# More restrictive (top 50%)
params = VLLM.sampling_params!(top_p: 0.5, temperature: 0.7)
```

## Top-k Sampling

Limits sampling to the top `k` most likely tokens.

```elixir
# Only consider top 50 tokens
params = VLLM.sampling_params!(top_k: 50, temperature: 0.7)

# Very restrictive (top 10 tokens)
params = VLLM.sampling_params!(top_k: 10, temperature: 0.7)

# Disable (default)
params = VLLM.sampling_params!(top_k: -1, temperature: 0.7)
```

## Token Limits

Control the length of generated text.

```elixir
params = VLLM.sampling_params!(
  max_tokens: 100,    # Maximum tokens to generate
  min_tokens: 10      # Minimum tokens (prevents very short outputs)
)
```

## Stop Sequences

Define strings or token IDs that stop generation.

```elixir
# Stop at newline or specific phrases
params = VLLM.sampling_params!(
  max_tokens: 200,
  stop: ["\\n", "END", "---"]
)

# Stop at specific token IDs
params = VLLM.sampling_params!(
  max_tokens: 200,
  stop_token_ids: [50256]  # EOS token for some models
)
```

## Repetition Control

Prevent repetitive text with penalties.

```elixir
params = VLLM.sampling_params!(
  # Penalize tokens that have appeared (reduces repetition)
  presence_penalty: 0.5,     # Range: -2.0 to 2.0

  # Penalize based on frequency of appearance
  frequency_penalty: 0.5,    # Range: -2.0 to 2.0

  # Multiplicative penalty for repeated tokens
  repetition_penalty: 1.1    # > 1.0 reduces repetition
)
```

## Multiple Completions

Generate multiple outputs for the same prompt.

```elixir
# Generate 3 completions
params = VLLM.sampling_params!(
  n: 3,
  temperature: 0.8,
  max_tokens: 50
)

outputs = VLLM.generate!(llm, prompt, sampling_params: params)
output = Enum.at(outputs, 0)

# Access all completions
completions = VLLM.attr!(output, "outputs")
Enum.each(completions, fn comp ->
  IO.puts(VLLM.attr!(comp, "text"))
end)
```

## Best-of Sampling

Generate multiple sequences and return the best.

```elixir
# Generate 5 sequences, return the best one
params = VLLM.sampling_params!(
  n: 1,
  best_of: 5,
  temperature: 0.8,
  max_tokens: 50
)
```

## Reproducibility

Use a seed for reproducible outputs.

```elixir
params = VLLM.sampling_params!(
  seed: 42,
  temperature: 0.7,
  max_tokens: 50
)

# Same seed + same prompt = same output
```

## All Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Randomness (0 = deterministic) |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `top_k` | int | -1 | Top-k sampling (-1 = disabled) |
| `max_tokens` | int | 16 | Maximum tokens to generate |
| `min_tokens` | int | 0 | Minimum tokens to generate |
| `presence_penalty` | float | 0.0 | Penalty for token presence |
| `frequency_penalty` | float | 0.0 | Penalty for token frequency |
| `repetition_penalty` | float | 1.0 | Multiplicative repetition penalty |
| `stop` | list | nil | Stop strings |
| `stop_token_ids` | list | nil | Stop token IDs |
| `n` | int | 1 | Number of completions |
| `best_of` | int | nil | Generate N, return best |
| `seed` | int | nil | Random seed |

## Recommended Settings

### Creative Writing
```elixir
VLLM.sampling_params!(temperature: 0.9, top_p: 0.95, max_tokens: 500)
```

### Factual Q&A
```elixir
VLLM.sampling_params!(temperature: 0.3, top_p: 0.9, max_tokens: 200)
```

### Code Generation
```elixir
VLLM.sampling_params!(temperature: 0.2, top_p: 0.95, max_tokens: 500, stop: ["```"])
```

### Chat/Conversation
```elixir
VLLM.sampling_params!(temperature: 0.7, top_p: 0.9, max_tokens: 300, repetition_penalty: 1.1)
```
