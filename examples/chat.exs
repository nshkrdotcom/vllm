# Chat Completions Example
#
# This example demonstrates the chat interface:
# - Creating conversation messages
# - Using system prompts
# - Multi-turn conversations
#
# IMPORTANT: vLLM requires a CUDA-capable GPU. If you don't have one,
# this example will fail.
#
# Run: mix run examples/chat.exs
#
# Note: Requires an instruction-tuned model (e.g., Qwen2 0.5B Instruct)

IO.puts("=== Chat Completions Example ===\n")
IO.puts("Note: vLLM requires a CUDA-capable GPU.")
IO.puts("If you see CUDA errors, ensure you have a compatible NVIDIA GPU.\n")

get_attr = fn ref, attr ->
  case SnakeBridge.Runtime.get_attr(ref, attr) do
    {:ok, value} -> value
    {:error, error} -> raise RuntimeError, message: "Failed to read #{attr}: #{inspect(error)}"
  end
end

Snakepit.run_as_script(fn ->
  # Load an instruction-tuned model
  # Note: Replace with a model you have access to
  IO.puts("Loading instruction-tuned model...")

  {:ok, llm} =
    Vllm.LLM.new("Qwen/Qwen2-0.5B-Instruct",
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
    Vllm.SamplingParams.new([], temperature: 0.7, max_tokens: 150, __runtime__: runtime_opts)

  # Example 1: Simple chat
  IO.puts("\n--- Simple Chat ---")

  messages1 = [
    [
      %{"role" => "system", "content" => "You are a helpful assistant."},
      %{"role" => "user", "content" => "What is the capital of Japan?"}
    ]
  ]

  {:ok, outputs} =
    Vllm.LLM.chat(llm, messages1, [], sampling_params: params, __runtime__: runtime_opts)

  output = Enum.at(outputs, 0)
  completion = get_attr.(output, :outputs) |> Enum.at(0)
  IO.puts("User: What is the capital of Japan?")
  IO.puts("Assistant: #{get_attr.(completion, :text)}")

  # Example 2: Chat with persona
  IO.puts("\n--- Chat with Persona ---")

  messages2 = [
    [
      %{
        "role" => "system",
        "content" => "You are a pirate who speaks in pirate dialect. Keep responses brief."
      },
      %{"role" => "user", "content" => "How do I learn to code?"}
    ]
  ]

  {:ok, outputs} =
    Vllm.LLM.chat(llm, messages2, [], sampling_params: params, __runtime__: runtime_opts)

  output = Enum.at(outputs, 0)
  completion = get_attr.(output, :outputs) |> Enum.at(0)
  IO.puts("User: How do I learn to code?")
  IO.puts("Pirate Assistant: #{get_attr.(completion, :text)}")

  # Example 3: Multi-turn conversation
  IO.puts("\n--- Multi-turn Conversation ---")

  messages3 = [
    [
      %{"role" => "system", "content" => "You are a math tutor. Be concise."},
      %{"role" => "user", "content" => "What is 15% of 80?"},
      %{"role" => "assistant", "content" => "15% of 80 is 12."},
      %{"role" => "user", "content" => "How did you calculate that?"}
    ]
  ]

  {:ok, outputs} =
    Vllm.LLM.chat(llm, messages3, [], sampling_params: params, __runtime__: runtime_opts)

  output = Enum.at(outputs, 0)
  completion = get_attr.(output, :outputs) |> Enum.at(0)
  IO.puts("User: What is 15% of 80?")
  IO.puts("Assistant: 15% of 80 is 12.")
  IO.puts("User: How did you calculate that?")
  IO.puts("Assistant: #{get_attr.(completion, :text)}")

  # Example 4: Batch chat (multiple conversations)
  IO.puts("\n--- Batch Chat (Multiple Conversations) ---")

  batch_messages = [
    [
      %{"role" => "user", "content" => "Say hello in French"}
    ],
    [
      %{"role" => "user", "content" => "Say hello in Spanish"}
    ],
    [
      %{"role" => "user", "content" => "Say hello in Japanese"}
    ]
  ]

  {:ok, short_params} =
    Vllm.SamplingParams.new([], temperature: 0.5, max_tokens: 30, __runtime__: runtime_opts)

  {:ok, outputs} =
    Vllm.LLM.chat(llm, batch_messages, [],
      sampling_params: short_params,
      __runtime__: runtime_opts
    )

  languages = ["French", "Spanish", "Japanese"]

  Enum.zip(outputs, languages)
  |> Enum.each(fn {output, lang} ->
    completion = get_attr.(output, :outputs) |> Enum.at(0)
    IO.puts("#{lang}: #{get_attr.(completion, :text)}")
  end)

  IO.puts("\nChat completions example complete!")
end)
