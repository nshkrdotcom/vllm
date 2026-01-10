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

VLLM.run(fn ->
  # Load an instruction-tuned model
  # Note: Replace with a model you have access to
  IO.puts("Loading instruction-tuned model...")

  llm =
    VLLM.llm!("Qwen/Qwen2-0.5B-Instruct",
      max_model_len: 2048,
      gpu_memory_utilization: 0.8
    )

  params = VLLM.sampling_params!(temperature: 0.7, max_tokens: 150)

  # Example 1: Simple chat
  IO.puts("\n--- Simple Chat ---")

  messages1 = [
    [
      %{"role" => "system", "content" => "You are a helpful assistant."},
      %{"role" => "user", "content" => "What is the capital of Japan?"}
    ]
  ]

  outputs = VLLM.chat!(llm, messages1, sampling_params: params)
  output = Enum.at(outputs, 0)
  completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
  IO.puts("User: What is the capital of Japan?")
  IO.puts("Assistant: #{VLLM.attr!(completion, "text")}")

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

  outputs = VLLM.chat!(llm, messages2, sampling_params: params)
  output = Enum.at(outputs, 0)
  completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
  IO.puts("User: How do I learn to code?")
  IO.puts("Pirate Assistant: #{VLLM.attr!(completion, "text")}")

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

  outputs = VLLM.chat!(llm, messages3, sampling_params: params)
  output = Enum.at(outputs, 0)
  completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
  IO.puts("User: What is 15% of 80?")
  IO.puts("Assistant: 15% of 80 is 12.")
  IO.puts("User: How did you calculate that?")
  IO.puts("Assistant: #{VLLM.attr!(completion, "text")}")

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

  short_params = VLLM.sampling_params!(temperature: 0.5, max_tokens: 30)
  outputs = VLLM.chat!(llm, batch_messages, sampling_params: short_params)

  languages = ["French", "Spanish", "Japanese"]

  Enum.zip(outputs, languages)
  |> Enum.each(fn {output, lang} ->
    completion = VLLM.attr!(output, "outputs") |> Enum.at(0)
    IO.puts("#{lang}: #{VLLM.attr!(completion, "text")}")
  end)

  IO.puts("\nChat completions example complete!")
end)
