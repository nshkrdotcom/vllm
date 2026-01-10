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

VLLM.run(fn ->
  IO.puts("Checking guided decoding support...")

  if VLLM.guided_decoding_supported?() do
    version = VLLM.version!()
    IO.puts("Guided decoding is available in vLLM #{version}.")

    IO.puts("\n--- Choice-Constrained Output ---")
    guided_choice = VLLM.guided_decoding_params!(choice: ["positive", "negative", "neutral"])
    IO.puts("Choice params: #{inspect(guided_choice)}")

    IO.puts("\n--- JSON Schema-Guided Output ---")
    json_schema = ~s({
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"}
      },
      "required": ["name", "age", "city"]
    })
    guided_json = VLLM.guided_decoding_params!(json: json_schema)
    IO.puts("JSON Schema: #{json_schema}")
    IO.puts("JSON params: #{inspect(guided_json)}")

    IO.puts("\n--- Regex-Constrained Output ---")
    phone_regex = "[0-9]{3}-[0-9]{3}-[0-9]{4}"
    guided_regex = VLLM.guided_decoding_params!(regex: phone_regex)
    IO.puts("Regex pattern: #{phone_regex}")
    IO.puts("Regex params: #{inspect(guided_regex)}")

    IO.puts("\n--- Email Pattern ---")
    email_regex = "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
    guided_email = VLLM.guided_decoding_params!(regex: email_regex)
    IO.puts("Email regex: #{email_regex}")
    IO.puts("Email params: #{inspect(guided_email)}")
  else
    version =
      case VLLM.version() do
        {:ok, value} -> value
        {:error, _} -> "unknown"
      end

    IO.puts("Guided decoding is not available in vLLM #{version}; skipping runtime demo.")

    IO.puts("\n--- Example Configurations ---")
    IO.puts("Choice: positive | negative | neutral")
    IO.puts("JSON schema: {name: string, age: integer, city: string}")
    IO.puts("Regex: [0-9]{3}-[0-9]{3}-[0-9]{4}")
    IO.puts("Email regex: [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
  end

  IO.puts("\nStructured output example complete!")
end)
