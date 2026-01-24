# Embeddings Example
#
# This example demonstrates using vLLM for embedding generation:
# - Loading an embedding model
# - Generating text embeddings
# - Batch embedding
#
# IMPORTANT: vLLM requires a CUDA-capable GPU.
#
# Run: mix run examples/embeddings.exs

IO.puts("=== Embeddings Example ===\n")
IO.puts("Note: vLLM requires a CUDA-capable GPU.\n")

{opts, _, _} = OptionParser.parse(System.argv(), switches: [model: :string])
model = opts[:model] || "BAAI/bge-small-en-v1.5"

texts = [
  "Elixir is a functional programming language.",
  "vLLM provides high-throughput inference.",
  "Embeddings turn text into vectors."
]

unwrap_embedding = fn
  [first | _] when is_list(first) -> first
  list -> list
end

get_attr = fn ref, attr ->
  case SnakeBridge.Runtime.get_attr(ref, attr) do
    {:ok, value} -> value
    {:error, reason} -> raise "Failed to read #{attr}: #{inspect(reason)}"
  end
end

extract_embedding = fn output ->
  extract_from_outputs = fn outputs ->
    cond do
      SnakeBridge.Ref.ref?(outputs) ->
        get_attr.(outputs, :embedding)

      is_map(outputs) ->
        Map.get(outputs, "embedding") || Map.get(outputs, :embedding) || outputs

      true ->
        outputs
    end
  end

  outputs =
    cond do
      SnakeBridge.Ref.ref?(output) ->
        try do
          get_attr.(output, :outputs)
        rescue
          _ -> output
        end

      is_map(output) ->
        Map.get(output, "outputs") || Map.get(output, :outputs) || output

      true ->
        output
    end

  outputs
  |> extract_from_outputs.()
  |> unwrap_embedding.()
end

IO.puts("Loading embedding model: #{model}")

Snakepit.run_as_script(fn ->
  {:ok, llm} =
    Vllm.LLM.new(model,
      runner: "pooling",
      dtype: "float16",
      gpu_memory_utilization: 0.8
    )

  llm_ref = SnakeBridge.Ref.from_wire_format(llm)

  runtime_opts =
    case llm_ref.pool_name do
      nil -> [session_id: llm_ref.session_id]
      pool_name -> [session_id: llm_ref.session_id, pool_name: pool_name]
    end

  {:ok, outputs} = Vllm.LLM.embed(llm, texts, __runtime__: runtime_opts)

  Enum.zip(texts, outputs)
  |> Enum.each(fn {text, output} ->
    vector = extract_embedding.(output)

    unless is_list(vector) do
      IO.puts("Unexpected embedding output: #{inspect(vector)}")
      System.halt(1)
    end

    preview =
      vector
      |> Enum.take(6)
      |> Enum.map(&Float.round(&1, 4))

    IO.puts("\nText: #{text}")
    IO.puts("Dimensions: #{length(vector)}")
    IO.puts("Preview: [#{Enum.join(preview, ", ")} ...]")
  end)

  IO.puts("\nEmbeddings example complete!")
end)
