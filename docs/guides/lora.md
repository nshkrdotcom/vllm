# LoRA Adapters

LoRA (Low-Rank Adaptation) enables efficient fine-tuning and serving of customized models.

## What is LoRA?

LoRA adds small trainable matrices to transformer layers while keeping base model weights frozen. This provides:

- **Efficient fine-tuning**: Train with less GPU memory
- **Small adapter files**: Typically 10-100 MB vs GB for full models
- **Easy deployment**: Switch adapters without reloading base model
- **Multi-adapter serving**: Serve different adapters from one base model

## Enabling LoRA

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
    enable_lora: true,
    max_lora_rank: 64,
    max_loras: 4
  )
end)
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `enable_lora` | Enable LoRA adapter support | false |
| `max_lora_rank` | Maximum adapter rank | 16 |
| `max_loras` | Maximum concurrent adapters | 1 |
| `lora_extra_vocab_size` | Extra vocabulary for adapters | 256 |

## Creating LoRA Requests

```elixir
# Create a LoRA request
lora = VLLM.lora_request!(
  "my-adapter",           # Unique name
  1,                      # Integer ID
  "/path/to/adapter"      # Path to adapter weights
)
```

## Using LoRA in Generation

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
    enable_lora: true,
    max_lora_rank: 64
  )

  # Create adapter request
  sql_lora = VLLM.lora_request!("sql-expert", 1, "/path/to/sql-adapter")

  params = VLLM.sampling_params!(temperature: 0.7, max_tokens: 200)

  # Generate with adapter
  outputs = VLLM.generate!(llm, "Write a SQL query to find all users",
    sampling_params: params,
    lora_request: sql_lora
  )
end)
```

## Multi-LoRA Serving

Serve different adapters for different requests:

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
    enable_lora: true,
    max_loras: 4
  )

  # Create multiple adapters
  sql_adapter = VLLM.lora_request!("sql", 1, "/adapters/sql")
  code_adapter = VLLM.lora_request!("code", 2, "/adapters/code")
  medical_adapter = VLLM.lora_request!("medical", 3, "/adapters/medical")

  params = VLLM.sampling_params!(max_tokens: 200)

  # Use different adapters per request
  VLLM.generate!(llm, "SQL query...", sampling_params: params, lora_request: sql_adapter)
  VLLM.generate!(llm, "Python function...", sampling_params: params, lora_request: code_adapter)
  VLLM.generate!(llm, "Medical diagnosis...", sampling_params: params, lora_request: medical_adapter)

  # Generate without adapter (base model)
  VLLM.generate!(llm, "General question...", sampling_params: params)
end)
```

## LoRA Adapter Format

vLLM expects LoRA adapters in the HuggingFace PEFT format:

```
adapter_directory/
├── adapter_config.json
├── adapter_model.bin (or .safetensors)
└── (optional) special_tokens_map.json
```

## Training LoRA Adapters

Popular tools for training LoRA adapters:

### HuggingFace PEFT

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

model = get_peft_model(base_model, config)
# Train...
model.save_pretrained("/path/to/adapter")
```

### LLaMA-Factory

```bash
python train.py --model_name meta-llama/Llama-2-7b-hf \
    --lora_rank 64 \
    --output_dir /path/to/adapter
```

## Performance Tips

1. **Adapter hot-swapping**: vLLM efficiently switches between adapters
2. **Batch different adapters**: Requests with different adapters can be batched
3. **Memory overhead**: Each loaded adapter adds minimal memory
4. **Rank trade-off**: Higher rank = more capacity but more memory

## Common Issues

### Adapter Not Loading

```elixir
# Check adapter path exists
# Check adapter_config.json is valid
# Ensure max_lora_rank >= adapter rank
```

### Memory Issues

```elixir
# Reduce max_loras
# Use smaller max_lora_rank
# Consider quantized base model
```

### Performance Issues

```elixir
# Batch requests with same adapter when possible
# Pre-load frequently used adapters
```

## Example: Task-Specific Adapters

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
    enable_lora: true,
    max_loras: 3
  )

  # Different adapters for different tasks
  adapters = %{
    summarization: VLLM.lora_request!("sum", 1, "/adapters/summarizer"),
    translation: VLLM.lora_request!("trans", 2, "/adapters/translator"),
    qa: VLLM.lora_request!("qa", 3, "/adapters/qa-expert")
  }

  # Route based on task
  task = :summarization
  adapter = Map.get(adapters, task)

  outputs = VLLM.generate!(llm, "Summarize this article...",
    lora_request: adapter
  )
end)
```
