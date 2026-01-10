# Basic vLLM Text Generation Example
#
# This example demonstrates the core VLLM functionality:
# - Creating an LLM instance
# - Generating text completions
# - Processing results
#
# IMPORTANT: vLLM requires a CUDA-capable GPU. If you don't have one,
# this example will fail. See the error handling below.
#
# Run: mix run examples/basic.exs

IO.puts("=== Basic vLLM Text Generation ===\n")

# Check for GPU requirement
IO.puts("Note: vLLM requires a CUDA-capable GPU.")
IO.puts("If you see CUDA errors, ensure you have a compatible NVIDIA GPU.\n")

VLLM.run(fn ->
  # Create an LLM instance with a small model
  # Note: facebook/opt-125m is used for demo purposes (small, fast)
  # For production, use larger models like Qwen/Qwen2-0.5B-Instruct
  IO.puts("Loading model: facebook/opt-125m")

  # Try loading with explicit settings
  llm =
    VLLM.llm!("facebook/opt-125m",
      dtype: "auto",
      gpu_memory_utilization: 0.8
    )

  # Define prompts to complete
  prompts = [
    "Hello, my name is",
    "The capital of France is",
    "Machine learning is"
  ]

  IO.puts("\nGenerating completions for #{length(prompts)} prompts...\n")

  # Generate completions
  outputs = VLLM.generate!(llm, prompts)

  # Process and display results
  Enum.each(outputs, fn output ->
    prompt = VLLM.attr!(output, "prompt")
    completions = VLLM.attr!(output, "outputs")
    first_completion = Enum.at(completions, 0)
    text = VLLM.attr!(first_completion, "text")

    IO.puts("Prompt: #{prompt}")
    IO.puts("Generated: #{text}")
    IO.puts("---")
  end)

  IO.puts("\nBasic generation complete!")
end)
