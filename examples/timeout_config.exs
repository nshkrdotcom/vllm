# Timeout Configuration Example
#
# This example demonstrates timeout configuration for vLLM operations:
# - Default ML inference timeout
# - Per-call timeout overrides
# - Timeout profiles
# - Timeout helper functions
#
# Run: mix run examples/timeout_config.exs

IO.puts("=== Timeout Configuration Example ===\n")

# Note: This example demonstrates the API without running actual inference
# to avoid long wait times in the demo

IO.puts("--- Timeout Profiles ---")
IO.puts("vLLM uses SnakeBridge's timeout architecture with these profiles:\n")
IO.puts("| Profile        | Timeout  | Use Case                    |")
IO.puts("|----------------|----------|------------------------------|")
IO.puts("| :default       | 2 min    | Standard Python calls        |")
IO.puts("| :streaming     | 30 min   | Streaming responses          |")
IO.puts("| :ml_inference  | 10 min   | LLM inference (VLLM default) |")
IO.puts("| :batch_job     | 1 hour   | Long-running batch operations|")

IO.puts("\n--- Using timeout_profile/1 ---")
profile_opts = VLLM.timeout_profile(:batch_job)
IO.puts("VLLM.timeout_profile(:batch_job)")
IO.puts("=> #{inspect(profile_opts)}")

IO.puts("\n--- Using timeout_ms/1 ---")
ms_opts = VLLM.timeout_ms(300_000)
IO.puts("VLLM.timeout_ms(300_000)")
IO.puts("=> #{inspect(ms_opts)}")

IO.puts("\n--- Using with_timeout/2 ---")

# Add timeout to empty options
opts1 = VLLM.with_timeout([], timeout: 60_000)
IO.puts("VLLM.with_timeout([], timeout: 60_000)")
IO.puts("=> #{inspect(opts1)}")

# Add timeout profile to existing options
opts2 = VLLM.with_timeout([sampling_params: :my_params], timeout_profile: :batch_job)
IO.puts("\nVLLM.with_timeout([sampling_params: :my_params], timeout_profile: :batch_job)")
IO.puts("=> #{inspect(opts2)}")

# Merge with existing runtime options
opts3 = VLLM.with_timeout([__runtime__: [verbose: true]], timeout: 120_000)
IO.puts("\nVLLM.with_timeout([__runtime__: [verbose: true]], timeout: 120_000)")
IO.puts("=> #{inspect(opts3)}")

IO.puts("\n--- Example Usage Patterns ---")

IO.puts("""

# 1. Quick inference with short timeout
VLLM.generate!(llm, prompt,
  sampling_params: params,
  __runtime__: [timeout: 30_000]  # 30 seconds
)

# 2. Batch job with long timeout
VLLM.generate!(llm, large_batch,
  Keyword.merge([sampling_params: params], VLLM.timeout_profile(:batch_job))
)

# 3. Using helper for cleaner code
opts = VLLM.with_timeout([sampling_params: params], timeout_profile: :ml_inference)
VLLM.generate!(llm, prompts, opts)

# 4. Per-call timeout in milliseconds
VLLM.generate!(llm, prompt,
  Keyword.merge([sampling_params: params], VLLM.timeout_ms(300_000))
)
""")

IO.puts("--- Configuration in config/config.exs ---")

IO.puts("""

# Set vLLM to use ML inference profile by default:
config :snakebridge,
  runtime: [
    library_profiles: %{"vllm" => :ml_inference}
  ]
""")

IO.puts("\nTimeout configuration example complete!")
