defmodule VLLM.ConfigHelperTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  alias VLLM.ConfigHelper

  setup do
    # Save and restore global app env touched by tests.
    prev_vllm = Application.get_all_env(:vllm)
    prev_snakepit = Application.get_all_env(:snakepit)
    prev_snakebridge = Application.get_all_env(:snakebridge)

    on_exit(fn ->
      restore_app_env(:vllm, prev_vllm)
      restore_app_env(:snakepit, prev_snakepit)
      restore_app_env(:snakebridge, prev_snakebridge)
    end)

    :ok
  end

  test "v1_multiprocessing :off forces VLLM_ENABLE_V1_MULTIPROCESSING=0" do
    Application.put_env(:vllm, :v1_multiprocessing, :off)

    :ok = ConfigHelper.configure_snakepit!()

    pool_config = Application.get_env(:snakepit, :pool_config)
    assert env_value(pool_config[:adapter_env], "VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
  end

  test "v1_multiprocessing :on sets worker multiproc method to spawn by default" do
    Application.put_env(:vllm, :v1_multiprocessing, :on)
    Application.put_env(:snakepit, :process_group_kill, true)

    if Snakepit.ProcessKiller.process_group_supported?() do
      :ok = ConfigHelper.configure_snakepit!()

      pool_config = Application.get_env(:snakepit, :pool_config)

      assert env_value(pool_config[:adapter_env], "VLLM_WORKER_MULTIPROC_METHOD") ==
               "spawn"
    else
      assert_raise ArgumentError, fn ->
        ConfigHelper.configure_snakepit!()
      end
    end
  end

  test "v1_multiprocessing respects preconfigured worker multiproc method" do
    Application.put_env(:vllm, :v1_multiprocessing, :on)
    Application.put_env(:snakepit, :process_group_kill, true)

    Application.put_env(:snakepit, :pool_config, %{
      adapter_env: %{"VLLM_WORKER_MULTIPROC_METHOD" => "fork"}
    })

    if Snakepit.ProcessKiller.process_group_supported?() do
      :ok = ConfigHelper.configure_snakepit!()

      pool_config = Application.get_env(:snakepit, :pool_config)

      assert env_value(pool_config[:adapter_env], "VLLM_WORKER_MULTIPROC_METHOD") ==
               "fork"
    else
      assert_raise ArgumentError, fn ->
        ConfigHelper.configure_snakepit!()
      end
    end
  end

  test "v1_multiprocessing :auto disables multiprocessing when process group kill is disabled" do
    Application.put_env(:vllm, :v1_multiprocessing, :auto)
    Application.put_env(:snakepit, :process_group_kill, false)

    capture_log(fn ->
      :ok = ConfigHelper.configure_snakepit!()
    end)

    pool_config = Application.get_env(:snakepit, :pool_config)
    assert env_value(pool_config[:adapter_env], "VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
  end

  test "v1_multiprocessing :on fails fast when process group kill is disabled" do
    Application.put_env(:vllm, :v1_multiprocessing, :on)
    Application.put_env(:snakepit, :process_group_kill, false)

    assert_raise ArgumentError, fn ->
      ConfigHelper.configure_snakepit!()
    end
  end

  test "configure_snakepit! updates multi-pool adapter_env in-place" do
    Application.put_env(:vllm, :v1_multiprocessing, :off)

    Application.put_env(:snakepit, :pools, [
      %{name: :default, adapter_env: %{"FOO" => "bar", "VLLM_ENABLE_V1_MULTIPROCESSING" => "1"}}
    ])

    :ok = ConfigHelper.configure_snakepit!()

    [pool] = Application.get_env(:snakepit, :pools)
    assert env_value(pool[:adapter_env], "FOO") == "bar"
    assert env_value(pool[:adapter_env], "VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
  end

  defp env_value(env, key) when is_map(env) do
    Map.get(env, key) || find_atom_key_value(env, key)
  end

  defp env_value(env, key) when is_list(env) do
    env
    |> Enum.find_value(fn
      {^key, value} -> value
      {k, value} -> if atom_key?(k, key), do: value, else: nil
      _ -> nil
    end)
  end

  defp env_value(_env, _key), do: nil

  defp find_atom_key_value(env, key) do
    Enum.find_value(env, fn
      {k, value} -> if atom_key?(k, key), do: value, else: nil
      _ -> nil
    end)
  end

  defp atom_key?(k, key) when is_atom(k), do: Atom.to_string(k) == key
  defp atom_key?(_k, _key), do: false

  defp restore_app_env(app, prev_env) when is_list(prev_env) do
    Application.delete_env(app, :__restore_marker__, persistent: true)

    # Clear all current keys then restore previous.
    for {key, _value} <- Application.get_all_env(app) do
      Application.delete_env(app, key)
    end

    for {key, value} <- prev_env do
      Application.put_env(app, key, value)
    end
  end
end
