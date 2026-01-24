defmodule VLLMTest do
  use ExUnit.Case, async: true

  # Ensure module is fully loaded for function_exported? checks
  setup_all do
    Code.ensure_loaded!(VLLM)
    :ok
  end

  describe "module structure" do
    test "exports run/1 and run/2" do
      assert function_exported?(VLLM, :run, 1)
      assert function_exported?(VLLM, :run, 2)
    end

    test "exports llm/1 and llm/2" do
      assert function_exported?(VLLM, :llm, 1)
      assert function_exported?(VLLM, :llm, 2)
    end

    test "exports llm!/1 and llm!/2" do
      assert function_exported?(VLLM, :llm!, 1)
      assert function_exported?(VLLM, :llm!, 2)
    end

    test "exports sampling_params/0 and sampling_params/1" do
      assert function_exported?(VLLM, :sampling_params, 0)
      assert function_exported?(VLLM, :sampling_params, 1)
    end

    test "exports sampling_params!/0 and sampling_params!/1" do
      assert function_exported?(VLLM, :sampling_params!, 0)
      assert function_exported?(VLLM, :sampling_params!, 1)
    end

    test "exports generate/2 and generate/3" do
      assert function_exported?(VLLM, :generate, 2)
      assert function_exported?(VLLM, :generate, 3)
    end

    test "exports generate!/2 and generate!/3" do
      assert function_exported?(VLLM, :generate!, 2)
      assert function_exported?(VLLM, :generate!, 3)
    end

    test "exports chat/2 and chat/3" do
      assert function_exported?(VLLM, :chat, 2)
      assert function_exported?(VLLM, :chat, 3)
    end

    test "exports chat!/2 and chat!/3" do
      assert function_exported?(VLLM, :chat!, 2)
      assert function_exported?(VLLM, :chat!, 3)
    end

    test "exports encode/2 and encode/3" do
      assert function_exported?(VLLM, :encode, 2)
      assert function_exported?(VLLM, :encode, 3)
    end

    test "exports engine/1 and engine/2" do
      assert function_exported?(VLLM, :engine, 1)
      assert function_exported?(VLLM, :engine, 2)
    end

    test "exports async_engine/1 and async_engine/2" do
      assert function_exported?(VLLM, :async_engine, 1)
      assert function_exported?(VLLM, :async_engine, 2)
    end

    test "exports pooling_params/0 and pooling_params/1" do
      assert function_exported?(VLLM, :pooling_params, 0)
      assert function_exported?(VLLM, :pooling_params, 1)
    end

    test "exports embed/2 and embed/3" do
      assert function_exported?(VLLM, :embed, 2)
      assert function_exported?(VLLM, :embed, 3)
    end

    test "exports lora_request/3 and lora_request/4" do
      assert function_exported?(VLLM, :lora_request, 3)
      assert function_exported?(VLLM, :lora_request, 4)
    end

    test "exports guided_decoding_params/0 and guided_decoding_params/1" do
      assert function_exported?(VLLM, :guided_decoding_params, 0)
      assert function_exported?(VLLM, :guided_decoding_params, 1)
    end
  end

  describe "timeout helpers" do
    test "with_timeout/2 adds __runtime__ option" do
      opts = VLLM.with_timeout([], timeout: 5000)
      assert opts == [__runtime__: [timeout: 5000]]
    end

    test "with_timeout/2 merges with existing __runtime__" do
      opts = VLLM.with_timeout([__runtime__: [foo: :bar]], timeout: 1000)
      assert opts == [__runtime__: [foo: :bar, timeout: 1000]]
    end

    test "with_timeout/2 preserves other options" do
      opts = VLLM.with_timeout([sampling_params: :some_params], timeout: 5000)
      assert Keyword.get(opts, :sampling_params) == :some_params
      assert Keyword.get(opts, :__runtime__) == [timeout: 5000]
    end

    test "with_timeout/2 can set timeout_profile" do
      opts = VLLM.with_timeout([], timeout_profile: :batch_job)
      assert opts == [__runtime__: [timeout_profile: :batch_job]]
    end

    test "timeout_profile/1 returns correct format for :default" do
      assert VLLM.timeout_profile(:default) == [__runtime__: [timeout_profile: :default]]
    end

    test "timeout_profile/1 returns correct format for :streaming" do
      assert VLLM.timeout_profile(:streaming) == [__runtime__: [timeout_profile: :streaming]]
    end

    test "timeout_profile/1 returns correct format for :ml_inference" do
      assert VLLM.timeout_profile(:ml_inference) == [
               __runtime__: [timeout_profile: :ml_inference]
             ]
    end

    test "timeout_profile/1 returns correct format for :batch_job" do
      assert VLLM.timeout_profile(:batch_job) == [__runtime__: [timeout_profile: :batch_job]]
    end

    test "timeout_profile/1 only accepts valid profiles" do
      assert_raise FunctionClauseError, fn ->
        VLLM.timeout_profile(:invalid)
      end
    end

    test "timeout_ms/1 returns correct format" do
      assert VLLM.timeout_ms(60_000) == [__runtime__: [timeout: 60_000]]
    end

    test "timeout_ms/1 requires positive integer" do
      assert_raise FunctionClauseError, fn ->
        VLLM.timeout_ms(0)
      end

      assert_raise FunctionClauseError, fn ->
        VLLM.timeout_ms(-100)
      end
    end

    test "timeout_ms/1 rejects non-integer" do
      assert_raise FunctionClauseError, fn ->
        VLLM.timeout_ms(60.5)
      end
    end
  end

  describe "FFI delegates" do
    test "exports call/2, call/3, call/4" do
      assert function_exported?(VLLM, :call, 2)
      assert function_exported?(VLLM, :call, 3)
      assert function_exported?(VLLM, :call, 4)
    end

    test "exports call!/2, call!/3, call!/4" do
      assert function_exported?(VLLM, :call!, 2)
      assert function_exported?(VLLM, :call!, 3)
      assert function_exported?(VLLM, :call!, 4)
    end

    test "exports get/2 and get!/2" do
      assert function_exported?(VLLM, :get, 2)
      assert function_exported?(VLLM, :get!, 2)
    end

    test "exports method/2, method/3, method/4" do
      assert function_exported?(VLLM, :method, 2)
      assert function_exported?(VLLM, :method, 3)
      assert function_exported?(VLLM, :method, 4)
    end

    test "exports method!/2, method!/3, method!/4" do
      assert function_exported?(VLLM, :method!, 2)
      assert function_exported?(VLLM, :method!, 3)
      assert function_exported?(VLLM, :method!, 4)
    end

    test "exports attr/2 and attr!/2" do
      assert function_exported?(VLLM, :attr, 2)
      assert function_exported?(VLLM, :attr!, 2)
    end

    test "exports set_attr/3" do
      assert function_exported?(VLLM, :set_attr, 3)
    end

    test "exports ref?/1" do
      assert function_exported?(VLLM, :ref?, 1)
    end

    test "exports bytes/1" do
      assert function_exported?(VLLM, :bytes, 1)
    end
  end

  describe "run/1 guard clause" do
    test "run/1 requires a function" do
      assert_raise FunctionClauseError, fn ->
        VLLM.run("not a function")
      end
    end

    test "run/1 requires a 0-arity function" do
      assert_raise FunctionClauseError, fn ->
        VLLM.run(fn _arg -> :ok end)
      end
    end
  end

  describe "generated wrappers" do
    test "Vllm module is loaded" do
      assert Code.ensure_loaded?(Vllm)
    end

    test "Vllm.LLM module is generated" do
      assert Code.ensure_loaded?(Vllm.LLM)
      assert function_exported?(Vllm.LLM, :new, 1)
      assert function_exported?(Vllm.LLM, :new, 2)
      assert function_exported?(Vllm.LLM, :generate, 3)
      assert function_exported?(Vllm.LLM, :chat, 3)
    end

    test "Vllm.SamplingParams module is generated" do
      assert Code.ensure_loaded?(Vllm.SamplingParams)
      assert function_exported?(Vllm.SamplingParams, :new, 1)
      assert function_exported?(Vllm.SamplingParams, :new, 2)
    end

    test "Vllm.PoolingParams module is generated" do
      assert Code.ensure_loaded?(Vllm.PoolingParams)
      assert function_exported?(Vllm.PoolingParams, :new, 1)
      assert function_exported?(Vllm.PoolingParams, :new, 2)
    end

    test "Vllm.LLMEngine module is generated" do
      assert Code.ensure_loaded?(Vllm.LLMEngine)
    end

    test "Vllm.AsyncLLMEngine module is generated" do
      assert Code.ensure_loaded?(Vllm.AsyncLLMEngine)
    end

    test "Vllm.RequestOutput module is generated" do
      assert Code.ensure_loaded?(Vllm.Outputs.RequestOutput)
    end

    test "Vllm.CompletionOutput module is generated" do
      assert Code.ensure_loaded?(Vllm.Outputs.CompletionOutput)
    end

    test "Vllm.Config.ModelConfig module is generated" do
      assert Code.ensure_loaded?(Vllm.Config.ModelConfig)
    end

    test "generated modules have correct snakebridge metadata" do
      assert Vllm.LLM.__snakebridge_library__() == "vllm"
      assert Vllm.LLM.__snakebridge_python_name__() == "vllm"
      assert Vllm.LLM.__snakebridge_python_class__() == "LLM"

      assert Vllm.SamplingParams.__snakebridge_library__() == "vllm"
      assert Vllm.SamplingParams.__snakebridge_python_class__() == "SamplingParams"
    end
  end
end
