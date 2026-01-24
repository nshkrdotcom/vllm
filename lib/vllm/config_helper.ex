defmodule VLLM.ConfigHelper do
  @moduledoc """
  Runtime configuration helper for using vLLM safely via SnakeBridge/Snakepit.

  This module composes `SnakeBridge.ConfigHelper` and adds a vLLM-specific
  safeguard around vLLM v1 multiprocessing (which can spawn child processes).

  ## v1 multiprocessing (vLLM 0.14+)

  vLLM can spawn subprocesses for the engine core. Under Snakepit, those
  subprocesses must be cleaned up deterministically on shutdown to avoid
  orphaned GPU processes holding memory.

  Configure with:

      config :vllm, v1_multiprocessing: :auto | :on | :off

  - `:off` - always forces `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  - `:on` - forces `VLLM_ENABLE_V1_MULTIPROCESSING=1` and fails fast unless
    Snakepit process-group cleanup is available
  - `:auto` - enables only when safe; otherwise forces off with a warning

  When v1 multiprocessing is enabled, this helper also sets
  `VLLM_WORKER_MULTIPROC_METHOD=spawn` (unless already provided) to avoid
  forking a multi-threaded Python gRPC server.
  """

  require Logger

  @type v1_multiprocessing_mode :: :auto | :on | :off

  @doc """
  Auto-configure Snakepit for SnakeBridge and enforce safe vLLM multiprocessing.

  Intended for `config/runtime.exs`:

      import Config
      VLLM.ConfigHelper.configure_snakepit!()

  Options:
  - all options supported by `SnakeBridge.ConfigHelper.configure_snakepit!/1`
  - `:v1_multiprocessing` - overrides `config :vllm, :v1_multiprocessing`
  """
  @spec configure_snakepit!(keyword()) :: :ok
  def configure_snakepit!(opts \\ []) do
    mode = v1_multiprocessing_mode!(opts)

    snakebridge_opts = Keyword.drop(opts, [:v1_multiprocessing])
    :ok = SnakeBridge.ConfigHelper.configure_snakepit!(snakebridge_opts)

    desired =
      case mode do
        :off ->
          "0"

        :on ->
          require_process_group_cleanup!()
          "1"

        :auto ->
          if process_group_cleanup_available?() do
            "1"
          else
            Logger.warning(
              "vLLM v1 multiprocessing disabled: Snakepit process-group cleanup is not available " <>
                "(enable with `config :snakepit, process_group_kill: true` on Unix with a fixed Snakepit, " <>
                "or set `config :vllm, v1_multiprocessing: :off` to silence this warning)."
            )

            "0"
          end
      end

    :ok = put_adapter_env!("VLLM_ENABLE_V1_MULTIPROCESSING", desired)

    if desired == "1" do
      :ok = put_adapter_env_unless_present!("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    end

    :ok
  end

  defp v1_multiprocessing_mode!(opts) do
    mode =
      Keyword.get(
        opts,
        :v1_multiprocessing,
        Application.get_env(:vllm, :v1_multiprocessing, :auto)
      )

    case mode do
      :auto ->
        :auto

      :on ->
        :on

      :off ->
        :off

      other ->
        raise ArgumentError,
              "invalid :v1_multiprocessing mode: #{inspect(other)} (expected :auto | :on | :off)"
    end
  end

  defp require_process_group_cleanup! do
    if process_group_cleanup_available?() do
      :ok
    else
      raise ArgumentError,
            "vLLM v1 multiprocessing requires Snakepit process-group cleanup, but it is not available. " <>
              "Enable it with `config :snakepit, process_group_kill: true` on Unix, and ensure you are using " <>
              "a Snakepit version that includes the pgid tracking fix."
    end
  end

  defp process_group_cleanup_available? do
    Application.get_env(:snakepit, :process_group_kill, true) and
      Code.ensure_loaded?(Snakepit.ProcessKiller) and
      function_exported?(Snakepit.ProcessKiller, :process_group_supported?, 0) and
      Snakepit.ProcessKiller.process_group_supported?() and
      Code.ensure_loaded?(Snakepit.Pool.ProcessRegistry) and
      function_exported?(Snakepit.Pool.ProcessRegistry, :update_process_group, 3)
  end

  defp put_adapter_env!(key, value) when is_binary(key) and is_binary(value) do
    pools = Application.get_env(:snakepit, :pools)
    pool_config = Application.get_env(:snakepit, :pool_config)

    cond do
      is_list(pools) and pools != [] ->
        updated =
          Enum.map(pools, fn pool ->
            pool_map = normalize_pool(pool)

            env =
              pool_map
              |> Map.get(:adapter_env)
              |> normalize_env_map()
              |> Map.put(key, value)

            Map.put(pool_map, :adapter_env, env)
          end)

        Application.put_env(:snakepit, :pools, updated)
        :ok

      is_map(pool_config) ->
        env =
          pool_config
          |> Map.get(:adapter_env)
          |> normalize_env_map()
          |> Map.put(key, value)

        Application.put_env(:snakepit, :pool_config, Map.put(pool_config, :adapter_env, env))
        :ok

      is_list(pool_config) ->
        pool_config
        |> Map.new()
        |> then(fn map -> Application.put_env(:snakepit, :pool_config, map) end)

        put_adapter_env!(key, value)

      true ->
        raise ArgumentError,
              "unable to configure Snakepit adapter_env: expected :snakepit :pools (non-empty list) " <>
                "or :snakepit :pool_config (map) to be set"
    end
  end

  defp normalize_pool(pool) when is_map(pool), do: pool
  defp normalize_pool(pool) when is_list(pool), do: Map.new(pool)

  defp normalize_env_map(nil), do: %{}

  defp normalize_env_map(env) when is_map(env) do
    Enum.reduce(env, %{}, fn {k, v}, acc ->
      Map.put(acc, safe_string(k), safe_string(v))
    end)
  end

  defp normalize_env_map(env) when is_list(env) do
    Enum.reduce(env, %{}, fn
      {k, v}, acc ->
        Map.put(acc, safe_string(k), safe_string(v))

      k, acc when is_binary(k) ->
        Map.put(acc, k, "")

      k, acc when is_atom(k) ->
        Map.put(acc, Atom.to_string(k), "")

      _, acc ->
        acc
    end)
  end

  defp put_adapter_env_unless_present!(key, value) when is_binary(key) and is_binary(value) do
    pools = Application.get_env(:snakepit, :pools)
    pool_config = Application.get_env(:snakepit, :pool_config)

    cond do
      is_list(pools) and pools != [] ->
        updated =
          Enum.map(pools, fn pool ->
            pool_map = normalize_pool(pool)

            env =
              pool_map
              |> Map.get(:adapter_env)
              |> normalize_env_map()
              |> ensure_env_key(key, value)

            Map.put(pool_map, :adapter_env, env)
          end)

        Application.put_env(:snakepit, :pools, updated)
        :ok

      is_map(pool_config) ->
        env =
          pool_config
          |> Map.get(:adapter_env)
          |> normalize_env_map()
          |> ensure_env_key(key, value)

        Application.put_env(:snakepit, :pool_config, Map.put(pool_config, :adapter_env, env))
        :ok

      is_list(pool_config) ->
        pool_config
        |> Map.new()
        |> then(fn map -> Application.put_env(:snakepit, :pool_config, map) end)

        put_adapter_env_unless_present!(key, value)

      true ->
        raise ArgumentError,
              "unable to configure Snakepit adapter_env: expected :snakepit :pools (non-empty list) " <>
                "or :snakepit :pool_config (map) to be set"
    end
  end

  defp ensure_env_key(env, key, value) do
    key = safe_string(key)
    value = safe_string(value)

    if Map.has_key?(env, key) do
      env
    else
      Map.put(env, key, value)
    end
  end

  defp safe_string(term) do
    to_string(term)
  rescue
    Protocol.UndefinedError -> inspect(term)
  end
end
