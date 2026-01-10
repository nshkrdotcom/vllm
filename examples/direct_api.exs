# Direct API / Universal FFI Example
#
# This example demonstrates direct access to vLLM's Python API:
# - Using VLLM.call/4 for any Python class/function
# - Using VLLM.method/4 for object methods
# - Using VLLM.attr/2 for attributes
#
# Run: mix run examples/direct_api.exs

IO.puts("=== Direct API / Universal FFI Example ===\n")

VLLM.run(fn ->
  # Example 1: Direct class instantiation
  IO.puts("--- Direct Class Instantiation ---")

  # Create SamplingParams directly
  params =
    VLLM.call!("vllm", "SamplingParams", [],
      temperature: 0.8,
      max_tokens: 50
    )

  IO.puts("Created SamplingParams via VLLM.call!/4")
  IO.puts("Type: #{inspect(params)}")

  # Example 2: Accessing module-level attributes
  IO.puts("\n--- Module Attributes ---")

  # Get vLLM version
  version = VLLM.get!("vllm", "__version__")
  IO.puts("vLLM version: #{version}")

  # Example 3: Working with objects
  IO.puts("\n--- Object Methods and Attributes ---")

  llm = VLLM.llm!("facebook/opt-125m")

  # Access LLM attributes
  # Note: Available attributes depend on vLLM version
  IO.puts("LLM object created")

  # Example 4: Calling arbitrary Python code
  IO.puts("\n--- Arbitrary Python Calls ---")

  # Import and use Python's os module
  cwd = VLLM.call!("os", "getcwd", [])
  IO.puts("Python current directory: #{cwd}")

  # Get environment variable
  home = VLLM.call!("os", "getenv", ["HOME"])
  IO.puts("HOME: #{home}")

  # Example 5: Working with Python builtins
  IO.puts("\n--- Python Builtins ---")

  # Using len()
  list = [1, 2, 3, 4, 5]
  len = VLLM.call!("builtins", "len", [list])
  IO.puts("len([1,2,3,4,5]) = #{len}")

  # Using str()
  str = VLLM.call!("builtins", "str", [42])
  IO.puts("str(42) = #{str}")

  # Example 6: Error handling
  IO.puts("\n--- Error Handling ---")

  # Non-bang version returns {:ok, value} or {:error, reason}
  case VLLM.call("vllm", "SamplingParams", [], temperature: 0.5) do
    {:ok, _params} -> IO.puts("SamplingParams created successfully")
    {:error, reason} -> IO.puts("Error: #{inspect(reason)}")
  end

  # Example 7: Type checking
  IO.puts("\n--- Type Checking ---")

  is_ref = VLLM.ref?(llm)
  IO.puts("Is llm a Python reference? #{is_ref}")

  is_ref_list = VLLM.ref?([1, 2, 3])
  IO.puts("Is [1,2,3] a Python reference? #{is_ref_list}")

  # Example 8: Generate with direct method call
  IO.puts("\n--- Direct Method Calls ---")

  prompts = ["The future of AI is"]
  outputs = VLLM.method!(llm, "generate", [prompts], sampling_params: params)

  output = Enum.at(outputs, 0)
  prompt = VLLM.attr!(output, "prompt")
  completions = VLLM.attr!(output, "outputs")
  first = Enum.at(completions, 0)
  text = VLLM.attr!(first, "text")

  IO.puts("Prompt: #{prompt}")
  IO.puts("Generated: #{text}")

  IO.puts("\nDirect API example complete!")
end)
