# LoRA Adapters Example
#
# This example demonstrates using LoRA adapters with vLLM:
# - Loading base model with LoRA support
# - Creating LoRA requests
# - Multi-LoRA serving
#
# Run: mix run examples/lora.exs
#
# Note: Requires LoRA adapter weights

IO.puts("=== LoRA Adapters Example ===\n")

IO.puts("--- What is LoRA? ---")

IO.puts("""
LoRA (Low-Rank Adaptation) enables efficient fine-tuning by:
- Adding small trainable matrices to transformer layers
- Keeping base model weights frozen
- Reducing memory and compute requirements significantly

Benefits:
- Train custom behaviors without full fine-tuning
- Serve multiple adapters from a single base model
- Switch between adapters per-request
""")

IO.puts("--- Example: Loading Model with LoRA Support ---")

IO.puts("""
# Enable LoRA when creating the LLM
llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
  enable_lora: true,
  max_lora_rank: 64,
  max_loras: 4
)
""")

IO.puts("--- Example: Creating LoRA Requests ---")

IO.puts("""
# Create LoRA request for a specific adapter
lora_sql = VLLM.lora_request!("sql-expert", 1, "/path/to/sql-lora-adapter")
lora_code = VLLM.lora_request!("code-expert", 2, "/path/to/code-lora-adapter")

# Use LoRA adapter in generation
outputs = VLLM.generate!(llm, "Write a SQL query to",
  sampling_params: params,
  lora_request: lora_sql
)
""")

IO.puts("--- Example: Multi-LoRA Serving ---")

IO.puts("""
# Serve different adapters for different requests
requests = [
  {"Write SQL to find all users", lora_sql},
  {"Write Python to sort a list", lora_code},
  {"Explain machine learning", nil}  # No adapter (base model)
]

# Process each with appropriate adapter
Enum.each(requests, fn {prompt, lora} ->
  opts = if lora, do: [lora_request: lora], else: []
  outputs = VLLM.generate!(llm, prompt, Keyword.merge([sampling_params: params], opts))
  # Process outputs...
end)
""")

IO.puts("--- LoRA Configuration Options ---")

IO.puts("""
| Option          | Description                           |
|-----------------|---------------------------------------|
| enable_lora     | Enable LoRA adapter support           |
| max_lora_rank   | Maximum rank of LoRA adapters (8-64)  |
| max_loras       | Maximum concurrent LoRA adapters      |
| lora_extra_vocab_size | Extra vocab for LoRA adapters  |
""")

IO.puts("--- Training LoRA Adapters ---")

IO.puts("""
LoRA adapters can be trained using:
- Hugging Face PEFT library
- LLaMA-Factory
- Axolotl
- Custom training scripts

The adapter weights are typically small (10-100 MB) compared to
the full model (GB-scale).
""")

IO.puts("\n--- Note ---")
IO.puts("Actual LoRA usage requires trained adapter weights.")
IO.puts("See vLLM documentation for LoRA adapter format requirements.")

IO.puts("\nLoRA example complete!")
