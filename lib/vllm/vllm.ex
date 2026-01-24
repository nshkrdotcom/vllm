defmodule VLLM do
  @moduledoc """
  VLLM - vLLM for Elixir via SnakeBridge.

  Easy, fast, and cheap LLM serving for everyone. This library provides
  transparent access to Python vLLM through SnakeBridge's generated wrappers.

  ## Quick Start

      VLLM.run(fn ->
        # Create an LLM instance
        llm = VLLM.llm!("facebook/opt-125m")

        # Generate text
        outputs = VLLM.generate!(llm, ["Hello, my name is"])

        # Process results
        Enum.each(outputs, fn output ->
          prompt = VLLM.attr!(output, "prompt")
          generated = VLLM.attr!(output, "outputs") |> Enum.at(0)
          text = VLLM.attr!(generated, "text")
          IO.puts("Prompt: \#{prompt}")
          IO.puts("Generated: \#{text}")
        end)
      end)

  ## Chat Interface

      VLLM.run(fn ->
        llm = VLLM.llm!("Qwen/Qwen2-0.5B-Instruct")

        messages = [[
          %{"role" => "system", "content" => "You are a helpful assistant."},
          %{"role" => "user", "content" => "What is the capital of France?"}
        ]]

        outputs = VLLM.chat!(llm, messages)
        # Process chat outputs...
      end)

  ## Sampling Parameters

  Control generation with `VLLM.SamplingParams`:

      VLLM.run(fn ->
        llm = VLLM.llm!("facebook/opt-125m")
        params = VLLM.sampling_params!(temperature: 0.8, top_p: 0.95, max_tokens: 100)

        outputs = VLLM.generate!(llm, ["Once upon a time"], sampling_params: params)
      end)

  ## Generated Wrappers

  This library uses SnakeBridge's generated wrappers for type-safe bindings:

    * `Vllm.LLM` - Main inference class
    * `Vllm.SamplingParams` - Generation parameters
    * `Vllm.PoolingParams` - Embedding parameters
    * `Vllm.LLMEngine` - Low-level engine
    * `Vllm.AsyncLLMEngine` - Async engine for serving

  ## Timeout Configuration

  VLLM leverages SnakeBridge's timeout architecture for LLM workloads.
  By default, all vLLM calls use the `:ml_inference` profile (10 minute timeout).

  ### Timeout Profiles

  | Profile         | Timeout  | Use Case                              |
  |-----------------|----------|---------------------------------------|
  | `:default`      | 2 min    | Standard Python calls                 |
  | `:streaming`    | 30 min   | Streaming responses                   |
  | `:ml_inference` | 10 min   | LLM inference (VLLM default)          |
  | `:batch_job`    | 1 hour   | Long-running batch operations         |

  ### Per-Call Timeout Override

      VLLM.generate!(llm, prompts,
        sampling_params: params,
        __runtime__: [timeout_profile: :batch_job]
      )

  ## Architecture

  VLLM uses SnakeBridge's generated wrappers to call vLLM:

      Elixir (VLLM module)
          |
      Generated Wrappers (Vllm.LLM, etc.)
          |
      SnakeBridge.Runtime
          |
      Snakepit gRPC
          |
      Python vLLM
          |
      GPU/TPU Inference

  All Python lifecycle is managed automatically by Snakepit.
  """

  # ---------------------------------------------------------------------------
  # Lifecycle Management
  # ---------------------------------------------------------------------------

  @doc """
  Run VLLM code with automatic Python lifecycle management.

  Wraps your code in `Snakepit.run_as_script/2` which:
  - Starts the Python process pool
  - Runs your code
  - Cleans up on exit

  Pass `halt: true` in opts if you need to force the BEAM to exit
  (for example, when running inside wrapper scripts).

  ## Example

      VLLM.run(fn ->
        llm = VLLM.llm!("facebook/opt-125m")
        outputs = VLLM.generate!(llm, ["Hello, world"])
        # ... process outputs
      end)
  """
  def run(fun, opts \\ []) when is_function(fun, 0) do
    Snakepit.run_as_script(fun, opts)
  end

  # ---------------------------------------------------------------------------
  # Core LLM API (delegating to generated wrappers)
  # ---------------------------------------------------------------------------

  @doc """
  Create a vLLM LLM instance for offline inference.

  Delegates to `Vllm.LLM.new/2`.

  ## Options

  Common options passed as keyword arguments:
    * `:dtype` - Data type ("auto", "float16", "bfloat16", "float32")
    * `:tensor_parallel_size` - Number of GPUs for tensor parallelism
    * `:gpu_memory_utilization` - Fraction of GPU memory to use (0.0-1.0)
    * `:max_model_len` - Maximum sequence length
    * `:quantization` - Quantization method ("awq", "gptq", "squeezellm", etc.)
    * `:trust_remote_code` - Whether to trust remote code from HuggingFace

  ## Examples

      {:ok, llm} = VLLM.llm("facebook/opt-125m")
      {:ok, llm} = VLLM.llm("Qwen/Qwen2-7B", tensor_parallel_size: 2)
      {:ok, llm} = VLLM.llm("TheBloke/Llama-2-7B-AWQ", quantization: "awq")
  """
  def llm(model, opts \\ []) do
    Vllm.LLM.new(model, opts)
  end

  @doc "Bang version of llm/2 - raises on error."
  def llm!(model, opts \\ []) do
    case llm(model, opts) do
      {:ok, llm} -> llm
      {:error, error} -> raise RuntimeError, message: "Failed to create LLM: #{inspect(error)}"
    end
  end

  @doc """
  Create SamplingParams for controlling text generation.

  Delegates to `Vllm.SamplingParams.new/2`.

  ## Options

    * `:temperature` - Sampling temperature (default: 1.0)
    * `:top_p` - Nucleus sampling probability (default: 1.0)
    * `:top_k` - Top-k sampling (default: -1, disabled)
    * `:max_tokens` - Maximum tokens to generate (default: 16)
    * `:min_tokens` - Minimum tokens to generate (default: 0)
    * `:presence_penalty` - Presence penalty (default: 0.0)
    * `:frequency_penalty` - Frequency penalty (default: 0.0)
    * `:repetition_penalty` - Repetition penalty (default: 1.0)
    * `:stop` - List of stop strings
    * `:stop_token_ids` - List of stop token IDs
    * `:n` - Number of completions to generate (default: 1)
    * `:best_of` - Number of sequences to generate and select best from
    * `:seed` - Random seed for reproducibility

  ## Examples

      {:ok, params} = VLLM.sampling_params(temperature: 0.8, max_tokens: 100)
      {:ok, params} = VLLM.sampling_params(top_p: 0.9, stop: ["\\n", "END"])
  """
  def sampling_params(opts \\ []) do
    Vllm.SamplingParams.new([], opts)
  end

  @doc "Bang version of sampling_params/1 - raises on error."
  def sampling_params!(opts \\ []) do
    case sampling_params(opts) do
      {:ok, params} ->
        params

      {:error, error} ->
        raise RuntimeError, message: "Failed to create SamplingParams: #{inspect(error)}"
    end
  end

  @doc """
  Generate text completions from prompts.

  Delegates to `Vllm.LLM.generate/4`.

  ## Arguments

    * `llm` - LLM instance from `VLLM.llm!/1`
    * `prompts` - String or list of strings to complete
    * `opts` - Options including:
      * `:sampling_params` - SamplingParams instance
      * `:use_tqdm` - Show progress bar (default: true)
      * `:lora_request` - LoRA adapter request

  ## Examples

      outputs = VLLM.generate!(llm, "Hello, my name is")
      outputs = VLLM.generate!(llm, ["Prompt 1", "Prompt 2"], sampling_params: params)

  ## Returns

  List of RequestOutput objects. Each has:
    * `prompt` - Original prompt
    * `outputs` - List of CompletionOutput objects
      * `text` - Generated text
      * `token_ids` - Generated token IDs
      * `finish_reason` - Reason for completion ("length", "stop", etc.)
  """
  def generate(llm, prompts, opts \\ []) do
    prompts = List.wrap(prompts)
    Vllm.LLM.generate(llm, prompts, opts)
  end

  @doc "Bang version of generate/3 - raises on error."
  def generate!(llm, prompts, opts \\ []) do
    case generate(llm, prompts, opts) do
      {:ok, outputs} -> outputs
      {:error, error} -> raise RuntimeError, message: "Generate failed: #{inspect(error)}"
    end
  end

  @doc """
  Generate chat completions from messages.

  Delegates to `Vllm.LLM.chat/4`.

  ## Arguments

    * `llm` - LLM instance from `VLLM.llm!/1`
    * `messages` - List of message conversations, where each conversation is a list of message maps
    * `opts` - Options including:
      * `:sampling_params` - SamplingParams instance
      * `:use_tqdm` - Show progress bar
      * `:chat_template` - Custom chat template (Jinja2 format)

  ## Message Format

  Each message is a map with:
    * `"role"` - One of "system", "user", "assistant"
    * `"content"` - Message content string

  ## Examples

      messages = [[
        %{"role" => "system", "content" => "You are helpful."},
        %{"role" => "user", "content" => "Hello!"}
      ]]

      outputs = VLLM.chat!(llm, messages)

  ## Returns

  List of RequestOutput objects (same as generate/3).
  """
  def chat(llm, messages, opts \\ []) do
    Vllm.LLM.chat(llm, messages, opts)
  end

  @doc "Bang version of chat/3 - raises on error."
  def chat!(llm, messages, opts \\ []) do
    case chat(llm, messages, opts) do
      {:ok, outputs} -> outputs
      {:error, error} -> raise RuntimeError, message: "Chat failed: #{inspect(error)}"
    end
  end

  @doc """
  Encode text to token IDs.

  Delegates to `Vllm.LLM.encode/3`.

  ## Examples

      {:ok, token_ids} = VLLM.encode(llm, "Hello, world!")
  """
  def encode(llm, text, opts \\ []) do
    Vllm.LLM.encode(llm, text, opts)
  end

  @doc "Bang version of encode/3."
  def encode!(llm, text, opts \\ []) do
    case encode(llm, text, opts) do
      {:ok, result} -> result
      {:error, error} -> raise RuntimeError, message: "Encode failed: #{inspect(error)}"
    end
  end

  # ---------------------------------------------------------------------------
  # Engine API (for advanced usage)
  # ---------------------------------------------------------------------------

  @doc """
  Create an LLMEngine for fine-grained control over inference.

  The LLMEngine provides lower-level access to vLLM's inference capabilities,
  useful for building custom serving solutions.

  Note: LLMEngine has a complex constructor requiring vllm_config and executor_class.
  This helper creates it from EngineArgs for simpler usage.

  ## Options

  Same as `llm/2` plus:
    * `:max_num_seqs` - Maximum number of sequences per batch
    * `:max_num_batched_tokens` - Maximum tokens per batch

  ## Examples

      {:ok, engine} = VLLM.engine("facebook/opt-125m")
  """
  def engine(model, opts \\ []) do
    # LLMEngine.from_engine_args is a class method, use SnakeBridge.call directly
    # First create EngineArgs, then use from_engine_args
    with {:ok, engine_args} <- SnakeBridge.call("vllm", "EngineArgs", [model], opts) do
      SnakeBridge.call("vllm", "LLMEngine.from_engine_args", [engine_args], opts)
    end
  end

  @doc "Bang version of engine/2."
  def engine!(model, opts \\ []) do
    case engine(model, opts) do
      {:ok, engine} ->
        engine

      {:error, error} ->
        raise RuntimeError, message: "Failed to create LLMEngine: #{inspect(error)}"
    end
  end

  @doc """
  Create an AsyncLLMEngine for asynchronous inference.

  Useful for building online serving applications with concurrent requests.

  ## Examples

      {:ok, engine} = VLLM.async_engine("facebook/opt-125m")
  """
  def async_engine(model, opts \\ []) do
    # AsyncLLMEngine.from_engine_args is a class method, use SnakeBridge.call directly
    with {:ok, engine_args} <- SnakeBridge.call("vllm", "EngineArgs", [model], opts) do
      SnakeBridge.call("vllm", "AsyncLLMEngine.from_engine_args", [engine_args], opts)
    end
  end

  @doc "Bang version of async_engine/2."
  def async_engine!(model, opts \\ []) do
    case async_engine(model, opts) do
      {:ok, engine} ->
        engine

      {:error, error} ->
        raise RuntimeError, message: "Failed to create AsyncLLMEngine: #{inspect(error)}"
    end
  end

  # ---------------------------------------------------------------------------
  # Pooling (Embeddings) API
  # ---------------------------------------------------------------------------

  @doc """
  Create PoolingParams for embedding models.

  Delegates to `Vllm.PoolingParams.new/2`.

  ## Options

    * `:additional_data` - Additional metadata for the pooling request

  ## Examples

      {:ok, params} = VLLM.pooling_params()
  """
  def pooling_params(opts \\ []) do
    Vllm.PoolingParams.new([], opts)
  end

  @doc "Bang version of pooling_params/1."
  def pooling_params!(opts \\ []) do
    case pooling_params(opts) do
      {:ok, params} ->
        params

      {:error, error} ->
        raise RuntimeError, message: "Failed to create PoolingParams: #{inspect(error)}"
    end
  end

  @doc """
  Generate embeddings for texts using a pooling model.

  Delegates to `Vllm.LLM.embed/3`.

  ## Arguments

    * `llm` - LLM instance configured with an embedding model
    * `texts` - String or list of strings to embed
    * `opts` - Options including:
      * `:pooling_params` - PoolingParams instance

  ## Examples

      llm = VLLM.llm!("intfloat/e5-mistral-7b-instruct", runner: "pooling")
      outputs = VLLM.embed!(llm, ["Hello, world!", "How are you?"])

  ## Returns

  List of EmbeddingRequestOutput objects with:
    * `outputs` - List of embeddings
  """
  def embed(llm, texts, opts \\ []) do
    texts = List.wrap(texts)
    Vllm.LLM.embed(llm, texts, opts)
  end

  @doc "Bang version of embed/3."
  def embed!(llm, texts, opts \\ []) do
    case embed(llm, texts, opts) do
      {:ok, outputs} -> outputs
      {:error, error} -> raise RuntimeError, message: "Embed failed: #{inspect(error)}"
    end
  end

  # ---------------------------------------------------------------------------
  # LoRA Support
  # ---------------------------------------------------------------------------

  @doc """
  Create a LoRARequest for serving LoRA adapters.

  ## Arguments

    * `name` - Unique name for this LoRA adapter
    * `lora_int_id` - Integer ID for the adapter
    * `lora_path` - Path to the LoRA adapter weights

  ## Examples

      {:ok, lora} = VLLM.lora_request("my-adapter", 1, "/path/to/adapter")
  """
  def lora_request(name, lora_int_id, lora_path, opts \\ []) do
    SnakeBridge.call("vllm.lora.request", "LoRARequest", [name, lora_int_id, lora_path], opts)
  end

  @doc "Bang version of lora_request/4."
  def lora_request!(name, lora_int_id, lora_path, opts \\ []) do
    SnakeBridge.call!("vllm.lora.request", "LoRARequest", [name, lora_int_id, lora_path], opts)
  end

  # ---------------------------------------------------------------------------
  # Guided Generation / Structured Outputs
  # ---------------------------------------------------------------------------

  @doc """
  Create guided decoding parameters for structured outputs.

  ## Options

    * `:json` - JSON schema string for JSON output
    * `:json_object` - Python dict/Pydantic model for JSON
    * `:regex` - Regex pattern for output
    * `:choice` - List of allowed string choices
    * `:grammar` - BNF grammar string

  ## Examples

      # JSON schema
      {:ok, guided} = VLLM.guided_decoding_params(
        json: ~s({"type": "object", "properties": {"name": {"type": "string"}}})
      )

      # Regex pattern
      {:ok, guided} = VLLM.guided_decoding_params(regex: "[0-9]{3}-[0-9]{4}")

      # Choice
      {:ok, guided} = VLLM.guided_decoding_params(choice: ["yes", "no", "maybe"])

  ## Support

  Guided decoding requires a vLLM build that exposes `GuidedDecodingParams`.
  Use `guided_decoding_supported?/0` to check availability.
  """
  def guided_decoding_params(opts \\ []) do
    if guided_decoding_supported?() do
      SnakeBridge.call("vllm", "GuidedDecodingParams", [], opts)
    else
      {:error, :guided_decoding_not_supported}
    end
  end

  @doc "Bang version of guided_decoding_params/1."
  def guided_decoding_params!(opts \\ []) do
    case guided_decoding_params(opts) do
      {:ok, guided} ->
        guided

      {:error, :guided_decoding_not_supported} ->
        version =
          case version() do
            {:ok, value} -> value
            {:error, _} -> "unknown"
          end

        raise ArgumentError,
              "Guided decoding is not available in vLLM #{version}. " <>
                "Upgrade vLLM or disable structured outputs."

      {:error, error} ->
        raise RuntimeError, message: "Guided decoding error: #{inspect(error)}"
    end
  end

  @doc """
  Check whether guided decoding parameters are available in the installed vLLM.
  """
  def guided_decoding_supported? do
    case SnakeBridge.get("vllm", "GuidedDecodingParams") do
      {:ok, _} -> true
      {:error, _} -> false
    end
  end

  # ---------------------------------------------------------------------------
  # Timeout helpers
  # ---------------------------------------------------------------------------

  @doc """
  Add timeout configuration to options.

  ## Options

    * `:timeout` - Exact timeout in milliseconds
    * `:timeout_profile` - Use a predefined profile

  ## Examples

      opts = VLLM.with_timeout([], timeout: 60_000)
      VLLM.generate!(llm, prompts, Keyword.merge(opts, sampling_params: params))
  """
  def with_timeout(opts, timeout_opts) when is_list(opts) and is_list(timeout_opts) do
    runtime = Keyword.get(opts, :__runtime__, [])
    new_runtime = Keyword.merge(runtime, timeout_opts)
    Keyword.put(opts, :__runtime__, new_runtime)
  end

  @doc """
  Timeout profile atoms for use with `__runtime__` option.

  ## Examples

      VLLM.generate!(llm, prompts,
        Keyword.merge([sampling_params: params], VLLM.timeout_profile(:batch_job))
      )
  """
  def timeout_profile(profile)
      when profile in [:default, :streaming, :ml_inference, :batch_job] do
    [__runtime__: [timeout_profile: profile]]
  end

  @doc """
  Create a timeout option for exact milliseconds.

  ## Examples

      VLLM.generate!(llm, prompts,
        Keyword.merge([sampling_params: params], VLLM.timeout_ms(300_000))
      )
  """
  def timeout_ms(milliseconds) when is_integer(milliseconds) and milliseconds > 0 do
    [__runtime__: [timeout: milliseconds]]
  end

  # ---------------------------------------------------------------------------
  # Universal FFI pass-through (convenience re-exports)
  # ---------------------------------------------------------------------------

  @doc """
  Call any vLLM function or class.

  ## Examples

      {:ok, result} = VLLM.call("vllm", "LLM", ["facebook/opt-125m"])
      {:ok, config} = VLLM.call("vllm.config", "ModelConfig", [], model: "...")
  """
  defdelegate call(module, function, args \\ [], opts \\ []), to: SnakeBridge

  @doc "Bang version - raises on error, returns value directly."
  defdelegate call!(module, function, args \\ [], opts \\ []), to: SnakeBridge

  @doc "Get a module attribute."
  defdelegate get(module, attr), to: SnakeBridge

  @doc "Bang version of get/2."
  defdelegate get!(module, attr), to: SnakeBridge

  @doc "Call a method on a Python object reference."
  defdelegate method(ref, method, args \\ [], opts \\ []), to: SnakeBridge

  @doc "Bang version of method/4."
  defdelegate method!(ref, method, args \\ [], opts \\ []), to: SnakeBridge

  @doc "Get an attribute from a Python object reference."
  defdelegate attr(ref, attribute), to: SnakeBridge

  @doc "Bang version of attr/2."
  defdelegate attr!(ref, attribute), to: SnakeBridge

  @doc "Set an attribute on a Python object reference."
  defdelegate set_attr(ref, attribute, value), to: SnakeBridge

  @doc "Check if a value is a Python object reference."
  defdelegate ref?(value), to: SnakeBridge

  @doc "Encode binary data as Python bytes."
  defdelegate bytes(data), to: SnakeBridge

  @doc "Get the installed vLLM version."
  def version do
    case SnakeBridge.get("vllm", "__version__") do
      {:ok, value} -> {:ok, to_string(value)}
      {:error, error} -> {:error, error}
    end
  end

  @doc "Bang version of version/0."
  def version! do
    case version() do
      {:ok, value} ->
        value

      {:error, error} ->
        raise RuntimeError, message: "Unable to read vLLM version: #{inspect(error)}"
    end
  end
end
