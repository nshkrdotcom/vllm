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
llm = VLLM.llm!("TheBloke/Llama-2-7B-AWQ",
  quantization: "awq",
  dtype: "auto"
)
""")

IO.puts("--- Example: Loading GPTQ Quantized Model ---")

IO.puts("""
# GPTQ quantized model
llm = VLLM.llm!("TheBloke/Llama-2-7B-GPTQ",
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
VLLM.run(fn ->
  IO.puts("\n--- Demo: Loading Standard Model ---")
  IO.puts("Loading facebook/opt-125m (small model for demo)...")

  format_error = fn error ->
    cond do
      is_exception(error) -> Exception.message(error)
      true -> inspect(error)
    end
  end

  result =
    with {:ok, llm} <- VLLM.llm("facebook/opt-125m", dtype: "auto"),
         {:ok, params} <- VLLM.sampling_params(temperature: 0.7, max_tokens: 30),
         {:ok, outputs} <- VLLM.generate(llm, "Quantization helps", sampling_params: params) do
      output = Enum.at(outputs, 0)
      completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
      IO.puts("Generated: Quantization helps#{VLLM.attr!(completion, "text")}")
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
