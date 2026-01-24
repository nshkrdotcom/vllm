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

Snakepit.run_as_script(fn ->
  get_attr = fn ref, attr ->
    case SnakeBridge.Runtime.get_attr(ref, attr) do
      {:ok, value} -> value
      {:error, reason} -> raise "Failed to read #{attr}: #{inspect(reason)}"
    end
  end

  {:ok, llm} =
    Vllm.LLM.new(model,
      max_model_len: 2048,
      gpu_memory_utilization: 0.8
    )

  llm_ref = SnakeBridge.Ref.from_wire_format(llm)

  runtime_opts =
    case llm_ref.pool_name do
      nil -> [session_id: llm_ref.session_id]
      pool_name -> [session_id: llm_ref.session_id, pool_name: pool_name]
    end

  {:ok, params} =
    Vllm.SamplingParams.new([], temperature: 0.2, max_tokens: 32, __runtime__: runtime_opts)

  IO.puts("\n--- Using timeout profile (:ml_inference) ---")

  {:ok, outputs} =
    Vllm.LLM.generate(llm, [prompt], [],
      sampling_params: params,
      __runtime__: Keyword.merge(runtime_opts, timeout_profile: :ml_inference)
    )

  output = Enum.at(outputs, 0)
  completion = get_attr.(output, :outputs) |> Enum.at(0)
  IO.puts("Response: #{get_attr.(completion, :text)}")

  IO.puts("\n--- Using explicit timeout (120_000 ms) ---")

  {:ok, outputs} =
    Vllm.LLM.generate(llm, [prompt], [],
      sampling_params: params,
      __runtime__: Keyword.merge(runtime_opts, timeout: 120_000)
    )

  output = Enum.at(outputs, 0)
  completion = get_attr.(output, :outputs) |> Enum.at(0)
  IO.puts("Response: #{get_attr.(completion, :text)}")

  IO.puts("\nTimeout configuration example complete!")
end)
