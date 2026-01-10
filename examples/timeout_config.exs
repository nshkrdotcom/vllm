# Timeout Configuration Example
#
# This example demonstrates timeout configuration for vLLM operations:
# - Timeout profiles
# - Per-call timeout overrides
# - Helper functions
#
# IMPORTANT: vLLM requires a CUDA-capable GPU.
#
# Run: mix run examples/timeout_config.exs

IO.puts("=== Timeout Configuration Example ===\n")
IO.puts("Note: vLLM requires a CUDA-capable GPU.\n")

{opts, _, _} =
  OptionParser.parse(System.argv(),
    switches: [model: :string, prompt: :string]
  )

model = opts[:model] || "facebook/opt-125m"
prompt = opts[:prompt] || "Explain Elixir in one sentence."

IO.puts("Loading model: #{model}")

VLLM.run(fn ->
  llm =
    VLLM.llm!(model,
      max_model_len: 2048,
      gpu_memory_utilization: 0.8
    )

  params = VLLM.sampling_params!(temperature: 0.2, max_tokens: 32)

  IO.puts("\n--- Using timeout profile (:ml_inference) ---")

  outputs =
    VLLM.generate!(
      llm,
      prompt,
      Keyword.merge([sampling_params: params], VLLM.timeout_profile(:ml_inference))
    )

  output = Enum.at(outputs, 0)
  completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
  IO.puts("Response: #{VLLM.attr!(completion, "text")}")

  IO.puts("\n--- Using explicit timeout (120_000 ms) ---")

  opts = VLLM.with_timeout([sampling_params: params], timeout: 120_000)
  outputs = VLLM.generate!(llm, prompt, opts)
  output = Enum.at(outputs, 0)
  completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
  IO.puts("Response: #{VLLM.attr!(completion, "text")}")

  IO.puts("\nTimeout configuration example complete!")
end)
