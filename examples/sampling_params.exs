# Sampling Parameters Example
#
# This example demonstrates how to control text generation using SamplingParams:
# - Temperature for randomness
# - Top-p (nucleus) sampling
# - Max tokens limit
# - Stop sequences
#
# IMPORTANT: vLLM requires a CUDA-capable GPU. If you don't have one,
# this example will fail.
#
# Run: mix run examples/sampling_params.exs

IO.puts("=== Sampling Parameters Example ===\n")
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

  prompt = "Once upon a time in a magical kingdom,"

  # Example 1: Low temperature (more deterministic)
  IO.puts("\n--- Low Temperature (0.3) ---")

  {:ok, params_low_temp} =
    Vllm.SamplingParams.new([], temperature: 0.3, max_tokens: 50, __runtime__: runtime_opts)

  {:ok, outputs} =
    Vllm.LLM.generate(llm, [prompt], [],
      sampling_params: params_low_temp,
      __runtime__: runtime_opts
    )

  output = Enum.at(outputs, 0)
  completion = get_attr.(output, :outputs) |> Enum.at(0)
  IO.puts("Prompt: #{prompt}")
  IO.puts("Generated: #{get_attr.(completion, :text)}")

  # Example 2: High temperature (more creative)
  IO.puts("\n--- High Temperature (0.9) ---")

  {:ok, params_high_temp} =
    Vllm.SamplingParams.new([], temperature: 0.9, max_tokens: 50, __runtime__: runtime_opts)

  {:ok, outputs} =
    Vllm.LLM.generate(llm, [prompt], [],
      sampling_params: params_high_temp,
      __runtime__: runtime_opts
    )

  output = Enum.at(outputs, 0)
  completion = get_attr.(output, :outputs) |> Enum.at(0)
  IO.puts("Prompt: #{prompt}")
  IO.puts("Generated: #{get_attr.(completion, :text)}")

  # Example 3: Top-p sampling
  IO.puts("\n--- Top-p Sampling (0.9) ---")

  {:ok, params_top_p} =
    Vllm.SamplingParams.new([],
      top_p: 0.9,
      temperature: 0.7,
      max_tokens: 50,
      __runtime__: runtime_opts
    )

  {:ok, outputs} =
    Vllm.LLM.generate(llm, [prompt], [], sampling_params: params_top_p, __runtime__: runtime_opts)

  output = Enum.at(outputs, 0)
  completion = get_attr.(output, :outputs) |> Enum.at(0)
  IO.puts("Prompt: #{prompt}")
  IO.puts("Generated: #{get_attr.(completion, :text)}")

  # Example 4: With stop sequences
  IO.puts("\n--- With Stop Sequences ---")

  params_stop =
    Vllm.SamplingParams.new([],
      temperature: 0.7,
      max_tokens: 100,
      stop: [".", "!", "?"],
      __runtime__: runtime_opts
    )

  {:ok, params_stop} = params_stop

  {:ok, outputs} =
    Vllm.LLM.generate(llm, [prompt], [], sampling_params: params_stop, __runtime__: runtime_opts)

  output = Enum.at(outputs, 0)
  completion = get_attr.(output, :outputs) |> Enum.at(0)
  finish_reason = get_attr.(completion, :finish_reason)
  IO.puts("Prompt: #{prompt}")
  IO.puts("Generated: #{get_attr.(completion, :text)}")
  IO.puts("Finish reason: #{finish_reason}")

  # Example 5: Multiple completions
  IO.puts("\n--- Multiple Completions (n=3) ---")

  {:ok, params_multi} =
    Vllm.SamplingParams.new([], temperature: 0.8, max_tokens: 30, n: 3, __runtime__: runtime_opts)

  {:ok, outputs} =
    Vllm.LLM.generate(llm, [prompt], [], sampling_params: params_multi, __runtime__: runtime_opts)

  output = Enum.at(outputs, 0)
  completions = get_attr.(output, :outputs)

  IO.puts("Prompt: #{prompt}")

  Enum.with_index(completions, 1)
  |> Enum.each(fn {comp, idx} ->
    IO.puts("  #{idx}. #{get_attr.(comp, :text)}")
  end)

  IO.puts("\nSampling parameters example complete!")
end)
