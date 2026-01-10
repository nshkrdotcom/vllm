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

VLLM.run(fn ->
  IO.puts("Loading model...")
  llm = VLLM.llm!("facebook/opt-125m")

  prompt = "Once upon a time in a magical kingdom,"

  # Example 1: Low temperature (more deterministic)
  IO.puts("\n--- Low Temperature (0.3) ---")
  params_low_temp = VLLM.sampling_params!(temperature: 0.3, max_tokens: 50)
  outputs = VLLM.generate!(llm, prompt, sampling_params: params_low_temp)
  output = Enum.at(outputs, 0)
  completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
  IO.puts("Prompt: #{prompt}")
  IO.puts("Generated: #{VLLM.attr!(completion, "text")}")

  # Example 2: High temperature (more creative)
  IO.puts("\n--- High Temperature (0.9) ---")
  params_high_temp = VLLM.sampling_params!(temperature: 0.9, max_tokens: 50)
  outputs = VLLM.generate!(llm, prompt, sampling_params: params_high_temp)
  output = Enum.at(outputs, 0)
  completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
  IO.puts("Prompt: #{prompt}")
  IO.puts("Generated: #{VLLM.attr!(completion, "text")}")

  # Example 3: Top-p sampling
  IO.puts("\n--- Top-p Sampling (0.9) ---")
  params_top_p = VLLM.sampling_params!(top_p: 0.9, temperature: 0.7, max_tokens: 50)
  outputs = VLLM.generate!(llm, prompt, sampling_params: params_top_p)
  output = Enum.at(outputs, 0)
  completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
  IO.puts("Prompt: #{prompt}")
  IO.puts("Generated: #{VLLM.attr!(completion, "text")}")

  # Example 4: With stop sequences
  IO.puts("\n--- With Stop Sequences ---")

  params_stop =
    VLLM.sampling_params!(
      temperature: 0.7,
      max_tokens: 100,
      stop: [".", "!", "?"]
    )

  outputs = VLLM.generate!(llm, prompt, sampling_params: params_stop)
  output = Enum.at(outputs, 0)
  completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
  finish_reason = VLLM.attr!(completion, "finish_reason")
  IO.puts("Prompt: #{prompt}")
  IO.puts("Generated: #{VLLM.attr!(completion, "text")}")
  IO.puts("Finish reason: #{finish_reason}")

  # Example 5: Multiple completions
  IO.puts("\n--- Multiple Completions (n=3) ---")
  params_multi = VLLM.sampling_params!(temperature: 0.8, max_tokens: 30, n: 3)
  outputs = VLLM.generate!(llm, prompt, sampling_params: params_multi)
  output = Enum.at(outputs, 0)
  completions = VLLM.attr!(output, "outputs")

  IO.puts("Prompt: #{prompt}")

  Enum.with_index(completions, 1)
  |> Enum.each(fn {comp, idx} ->
    IO.puts("  #{idx}. #{VLLM.attr!(comp, "text")}")
  end)

  IO.puts("\nSampling parameters example complete!")
end)
