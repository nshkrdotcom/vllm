# Online Serving

While VLLM for Elixir primarily focuses on offline inference, you can build online serving solutions using the Engine APIs.

## AsyncLLMEngine

For concurrent request handling:

```elixir
VLLM.run(fn ->
  # Create async engine
  engine = VLLM.async_engine!("facebook/opt-125m",
    gpu_memory_utilization: 0.9
  )

  # Handle concurrent requests
  # (Implementation depends on your serving requirements)
end)
```

## Building a Simple Server

You can wrap VLLM in a Phoenix or Plug-based HTTP server:

```elixir
defmodule MyApp.LLMServer do
  use GenServer

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    # Initialize in a task to not block
    Task.start(fn ->
      VLLM.run(fn ->
        llm = VLLM.llm!("facebook/opt-125m")
        # Store llm reference for handling requests
        # Note: This is a simplified example
      end)
    end)

    {:ok, %{}}
  end

  def generate(prompt, opts \\ []) do
    GenServer.call(__MODULE__, {:generate, prompt, opts})
  end
end
```

## OpenAI-Compatible Server

For full OpenAI API compatibility, consider running the Python vLLM server directly and calling it from Elixir via HTTP:

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \\
  --model facebook/opt-125m \\
  --port 8000
```

Then call from Elixir using any HTTP client:

```elixir
defmodule MyApp.VLLMClient do
  def chat_completion(messages, opts \\ []) do
    body = %{
      model: Keyword.get(opts, :model, "facebook/opt-125m"),
      messages: messages,
      temperature: Keyword.get(opts, :temperature, 0.7),
      max_tokens: Keyword.get(opts, :max_tokens, 100)
    }

    case HTTPoison.post("http://localhost:8000/v1/chat/completions",
                        Jason.encode!(body),
                        [{"Content-Type", "application/json"}]) do
      {:ok, %{status_code: 200, body: body}} ->
        {:ok, Jason.decode!(body)}
      {:error, reason} ->
        {:error, reason}
    end
  end
end
```

## Streaming Responses

For streaming, use the AsyncLLMEngine with iteration:

```elixir
# Note: Streaming implementation depends on how you want to
# deliver partial results to clients (WebSocket, SSE, etc.)
```

## Scaling Considerations

1. **Single Engine**: vLLM's continuous batching handles multiple requests efficiently
2. **Multiple Engines**: For very high throughput, run multiple engines on different GPUs
3. **Load Balancing**: Use nginx or similar for distributing requests across engines

## Timeout Configuration

For online serving, configure appropriate timeouts:

```elixir
# In config/config.exs
config :snakebridge,
  runtime: [
    library_profiles: %{"vllm" => :streaming}  # 30 min timeout
  ]

# Or per-request
outputs = VLLM.generate!(llm, prompt,
  __runtime__: [timeout_profile: :streaming]
)
```

## Health Checks

Implement health checks for your serving infrastructure:

```elixir
defmodule MyApp.HealthCheck do
  def check_vllm do
    try do
      VLLM.run(fn ->
        llm = VLLM.llm!("facebook/opt-125m")
        outputs = VLLM.generate!(llm, "test", sampling_params: VLLM.sampling_params!(max_tokens: 1))
        :ok
      end, timeout: 30_000)
    rescue
      _ -> :error
    end
  end
end
```
