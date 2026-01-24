# Quantization Example
#
# This example demonstrates using quantized models with vLLM:
# - AWQ quantization
# - GPTQ quantization
# - Memory-efficient inference
#
# Run: mix run examples/quantization.exs
#
# Note: Requires compatible quantized model weights

IO.puts("=== Quantization Example ===\n")

IO.puts("--- Supported Quantization Methods ---")

IO.puts("""
vLLM supports various quantization methods for memory-efficient inference:

| Method      | Description                              |
|-------------|------------------------------------------|
| AWQ         | Activation-aware Weight Quantization     |
| GPTQ        | Post-training quantization               |
| SqueezeLLM  | Sensitivity-based non-uniform quantization|
| FP8         | 8-bit floating point                     |
| INT4/INT8   | Integer quantization                     |
| BitsAndBytes| 4-bit quantization via bitsandbytes     |
""")

IO.puts("--- Example: Loading AWQ Quantized Model ---")

IO.puts("""
# AWQ quantized Llama 2 7B
{:ok, llm} = Vllm.LLM.new("TheBloke/Llama-2-7B-AWQ",
  quantization: "awq",
  dtype: "auto"
)
""")

IO.puts("--- Example: Loading GPTQ Quantized Model ---")

IO.puts("""
# GPTQ quantized model
{:ok, llm} = Vllm.LLM.new("TheBloke/Llama-2-7B-GPTQ",
  quantization: "gptq",
  dtype: "float16"
)
""")

IO.puts("--- Memory Comparison ---")

IO.puts("""
Model: Llama 2 7B

| Precision    | Memory (approx) | Throughput |
|--------------|-----------------|------------|
| FP32         | ~28 GB          | Baseline   |
| FP16/BF16    | ~14 GB          | ~1.5-2x    |
| INT8         | ~7 GB           | ~1.8-2.5x  |
| INT4 (AWQ)   | ~3.5 GB         | ~2-3x      |
| INT4 (GPTQ)  | ~3.5 GB         | ~2-3x      |
""")

# Demonstration run (with opt-125m for speed)
Snakepit.run_as_script(fn ->
  IO.puts("\n--- Demo: Loading Standard Model ---")
  IO.puts("Loading facebook/opt-125m (small model for demo)...")

  get_attr = fn ref, attr ->
    case SnakeBridge.Runtime.get_attr(ref, attr) do
      {:ok, value} -> value
      {:error, reason} -> raise "Failed to read #{attr}: #{inspect(reason)}"
    end
  end

  format_error = fn error ->
    cond do
      is_exception(error) -> Exception.message(error)
      true -> inspect(error)
    end
  end

  result =
    with {:ok, llm} <- Vllm.LLM.new("facebook/opt-125m", dtype: "auto"),
         llm_ref = SnakeBridge.Ref.from_wire_format(llm),
         runtime_opts <-
           (case llm_ref.pool_name do
              nil -> [session_id: llm_ref.session_id]
              pool_name -> [session_id: llm_ref.session_id, pool_name: pool_name]
            end),
         {:ok, params} <-
           Vllm.SamplingParams.new([],
             temperature: 0.7,
             max_tokens: 30,
             __runtime__: runtime_opts
           ),
         {:ok, outputs} <-
           Vllm.LLM.generate(llm, ["Quantization helps"], [],
             sampling_params: params,
             __runtime__: runtime_opts
           ) do
      output = Enum.at(outputs, 0)
      completion = get_attr.(output, :outputs) |> Enum.at(0)
      IO.puts("Generated: Quantization helps#{get_attr.(completion, :text)}")
      :ok
    end

  case result do
    :ok ->
      :ok

    {:error, error} ->
      IO.puts("Skipping demo due to error: #{format_error.(error)}")
  end

  IO.puts("\nQuantization example complete!")
end)
