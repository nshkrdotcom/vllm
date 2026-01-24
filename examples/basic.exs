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

get_attr = fn ref, attr ->
  case SnakeBridge.Runtime.get_attr(ref, attr) do
    {:ok, value} -> value
    {:error, error} -> raise RuntimeError, message: "Failed to read #{attr}: #{inspect(error)}"
  end
end

Snakepit.run_as_script(fn ->
  # Create an LLM instance with a small model
  # Note: facebook/opt-125m is used for demo purposes (small, fast)
  # For production, use larger models like Qwen/Qwen2-0.5B-Instruct
  IO.puts("Loading model: facebook/opt-125m")

  # Try loading with explicit settings
  {:ok, llm} =
    Vllm.LLM.new("facebook/opt-125m",
      dtype: "auto",
      gpu_memory_utilization: 0.8
    )

  llm_ref = SnakeBridge.Ref.from_wire_format(llm)

  runtime_opts =
    case llm_ref.pool_name do
      nil -> [session_id: llm_ref.session_id]
      pool_name -> [session_id: llm_ref.session_id, pool_name: pool_name]
    end

  # Define prompts to complete
  prompts = [
    "Hello, my name is",
    "The capital of France is",
    "Machine learning is"
  ]

  IO.puts("\nGenerating completions for #{length(prompts)} prompts...\n")

  # Generate completions
  {:ok, outputs} = Vllm.LLM.generate(llm, prompts, [], __runtime__: runtime_opts)

  # Process and display results
  Enum.each(outputs, fn output ->
    prompt = get_attr.(output, :prompt)
    completions = get_attr.(output, :outputs)
    first_completion = Enum.at(completions, 0)
    text = get_attr.(first_completion, :text)

    IO.puts("Prompt: #{prompt}")
    IO.puts("Generated: #{text}")
    IO.puts("---")
  end)

  IO.puts("\nBasic generation complete!")
end)
