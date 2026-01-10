# Multi-GPU / Tensor Parallelism Example
#
# This example demonstrates distributed inference with vLLM:
# - Tensor parallelism across multiple GPUs
# - Pipeline parallelism
# - Memory utilization settings
#
# IMPORTANT: vLLM requires CUDA-capable GPU(s). If you don't have one,
# this example will fail.
#
# Run: mix run examples/multi_gpu.exs
#
# Note: Requires multiple GPUs for actual parallel execution

IO.puts("=== Multi-GPU / Tensor Parallelism Example ===\n")
IO.puts("Note: vLLM requires CUDA-capable GPU(s).")
IO.puts("If you see CUDA errors, ensure you have compatible NVIDIA GPU(s).\n")

IO.puts("--- Parallelism Options ---")

IO.puts("""
vLLM supports multiple parallelism strategies:

| Strategy            | Use Case                              |
|---------------------|---------------------------------------|
| Tensor Parallelism  | Split model layers across GPUs        |
| Pipeline Parallelism| Split model stages across GPUs        |
| Data Parallelism    | Multiple model replicas               |
| Expert Parallelism  | MoE models across GPUs                |
""")

IO.puts("--- Example: Tensor Parallelism (2 GPUs) ---")

IO.puts("""
# Load large model across 2 GPUs
llm = VLLM.llm!("meta-llama/Llama-2-13b-hf",
  tensor_parallel_size: 2,
  gpu_memory_utilization: 0.9
)
""")

IO.puts("--- Example: Pipeline Parallelism ---")

IO.puts("""
# Pipeline parallelism for very large models
llm = VLLM.llm!("meta-llama/Llama-2-70b-hf",
  tensor_parallel_size: 4,
  pipeline_parallel_size: 2,
  gpu_memory_utilization: 0.95
)
""")

IO.puts("--- Memory Utilization ---")

IO.puts("""
The gpu_memory_utilization parameter controls how much GPU memory vLLM can use:

| Value | Description                              |
|-------|------------------------------------------|
| 0.5   | Conservative, leaves room for other apps |
| 0.8   | Balanced (default)                       |
| 0.9   | High utilization                         |
| 0.95  | Maximum, minimal overhead                |

Higher utilization = more KV cache = higher throughput
""")

IO.puts("--- Checking Available GPUs ---")

IO.puts("""
# In Python (via VLLM.call):
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")

# Or check CUDA_VISIBLE_DEVICES environment variable
""")

# Demo with single GPU
VLLM.run(fn ->
  IO.puts("\n--- Demo: Single GPU Configuration ---")
  IO.puts("Loading model with explicit GPU settings...")

  llm =
    VLLM.llm!("facebook/opt-125m",
      tensor_parallel_size: 1,
      gpu_memory_utilization: 0.8,
      dtype: "auto"
    )

  params = VLLM.sampling_params!(temperature: 0.7, max_tokens: 30)
  outputs = VLLM.generate!(llm, "Distributed computing", sampling_params: params)

  output = Enum.at(outputs, 0)
  completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
  IO.puts("Generated: Distributed computing#{VLLM.attr!(completion, "text")}")

  IO.puts("\nMulti-GPU example complete!")
end)
