# Batch Inference Example
#
# This example demonstrates vLLM's high-throughput batch processing:
# - Processing multiple prompts efficiently
# - Continuous batching for maximum throughput
# - Measuring performance
#
# IMPORTANT: vLLM requires a CUDA-capable GPU. If you don't have one,
# this example will fail.
#
# Run: mix run examples/batch_inference.exs

IO.puts("=== Batch Inference Example ===\n")
IO.puts("Note: vLLM requires a CUDA-capable GPU.")
IO.puts("If you see CUDA errors, ensure you have a compatible NVIDIA GPU.\n")

VLLM.run(fn ->
  IO.puts("Loading model...")
  llm = VLLM.llm!("facebook/opt-125m")

  # Create a batch of prompts
  prompts = [
    "The meaning of life is",
    "Artificial intelligence will",
    "The best programming language is",
    "In the year 2050,",
    "Climate change affects",
    "The secret to happiness is",
    "Technology has transformed",
    "Education in the future will",
    "Space exploration enables",
    "The internet has revolutionized"
  ]

  IO.puts("Processing #{length(prompts)} prompts in batch...\n")

  params = VLLM.sampling_params!(temperature: 0.7, max_tokens: 50)

  # Measure batch processing time
  start_time = System.monotonic_time(:millisecond)
  outputs = VLLM.generate!(llm, prompts, sampling_params: params)
  end_time = System.monotonic_time(:millisecond)

  elapsed_ms = end_time - start_time
  prompts_per_second = length(prompts) / (elapsed_ms / 1000)

  IO.puts("--- Results ---\n")

  Enum.each(outputs, fn output ->
    prompt = VLLM.attr!(output, "prompt")
    completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
    text = VLLM.attr!(completion, "text")

    # Truncate for display
    text_preview = String.slice(text, 0, 60)
    text_preview = if String.length(text) > 60, do: text_preview <> "...", else: text_preview

    IO.puts("#{prompt}#{text_preview}")
  end)

  IO.puts("\n--- Performance ---")
  IO.puts("Total prompts: #{length(prompts)}")
  IO.puts("Total time: #{elapsed_ms}ms")
  IO.puts("Throughput: #{Float.round(prompts_per_second, 2)} prompts/second")

  IO.puts("\nBatch inference example complete!")
end)
