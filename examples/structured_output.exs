# Structured Output Example
#
# This example demonstrates guided generation for structured outputs:
# - JSON schema-guided generation
# - Regex-constrained outputs
# - Choice-constrained outputs
#
# Run: mix run examples/structured_output.exs
#
# Note: Requires vLLM with structured output support

IO.puts("=== Structured Output Example ===\n")

Snakepit.run_as_script(fn ->
  version =
    case Vllm.CollectEnv.get_vllm_version() do
      {:ok, value} -> value
      {:error, _} -> "unknown"
    end

  IO.puts("Using vLLM #{version}.")
  IO.puts("Checking structured outputs via SamplingParams...")

  json_schema = ~s({
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "age": {"type": "integer"},
      "city": {"type": "string"}
    },
    "required": ["name", "age", "city"]
  })

  scenarios = [
    {"Choice-Constrained Output", %{choice: ["positive", "negative", "neutral"]}},
    {"JSON Schema-Guided Output", %{json: json_schema}},
    {"Regex-Constrained Output", %{regex: "[0-9]{3}-[0-9]{3}-[0-9]{4}"}},
    {"Email Pattern", %{regex: "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"}}
  ]

  Enum.each(scenarios, fn {title, structured_outputs} ->
    IO.puts("\n--- #{title} ---")

    case Vllm.SamplingParams.new([], structured_outputs: structured_outputs) do
      {:ok, params} ->
        case Vllm.SamplingParams.structured_outputs(params) do
          {:ok, value} ->
            IO.puts("Structured params: #{inspect(value)}")

          {:error, reason} ->
            IO.puts("Could not read structured outputs: #{inspect(reason)}")
        end

      {:error, reason} ->
        IO.puts("Structured outputs not available: #{inspect(reason)}")
    end
  end)

  IO.puts("\nStructured output example complete!")
end)
