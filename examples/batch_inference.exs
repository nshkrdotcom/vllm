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

get_attr = fn ref, attr ->
  case SnakeBridge.Runtime.get_attr(ref, attr) do
    {:ok, value} -> value
    {:error, error} -> raise RuntimeError, message: "Failed to read #{attr}: #{inspect(error)}"
  end
end

Snakepit.run_as_script(fn ->
  IO.puts("Loading model...")
  {:ok, llm} = Vllm.LLM.new("facebook/opt-125m")
  llm_ref = SnakeBridge.Ref.from_wire_format(llm)

  runtime_opts =
    case llm_ref.pool_name do
      nil -> [session_id: llm_ref.session_id]
      pool_name -> [session_id: llm_ref.session_id, pool_name: pool_name]
    end

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

  {:ok, params} =
    Vllm.SamplingParams.new([], temperature: 0.7, max_tokens: 50, __runtime__: runtime_opts)

  # Measure batch processing time
  start_time = System.monotonic_time(:millisecond)

  {:ok, outputs} =
    Vllm.LLM.generate(llm, prompts, [], sampling_params: params, __runtime__: runtime_opts)

  end_time = System.monotonic_time(:millisecond)

  elapsed_ms = end_time - start_time
  prompts_per_second = length(prompts) / (elapsed_ms / 1000)

  IO.puts("--- Results ---\n")

  Enum.each(outputs, fn output ->
    prompt = get_attr.(output, :prompt)
    completion = get_attr.(output, :outputs) |> Enum.at(0)
    text = get_attr.(completion, :text)

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
