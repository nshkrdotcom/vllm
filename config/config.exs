import Config

config :snakebridge,
  verbose: false,
  runtime: [
    library_profiles: %{"vllm" => :ml_inference}
  ]

# vLLM v1 multiprocessing spawns subprocesses; enable only when Snakepit can
# reliably kill the worker process group on shutdown.
config :vllm, v1_multiprocessing: :auto

# Compile-time snakepit config so SnakeBridge installs into the same venv
# used at runtime (can't call runtime config helpers in config.exs).
project_root = Path.expand("..", __DIR__)

snakebridge_venv =
  [
    System.get_env("SNAKEBRIDGE_VENV"),
    Path.join(project_root, ".venv"),
    Path.expand("../snakebridge/.venv", __DIR__)
  ]
  |> Enum.find(fn path -> is_binary(path) and File.dir?(path) end)

python_executable =
  if snakebridge_venv do
    [
      Path.join([snakebridge_venv, "bin", "python3"]),
      Path.join([snakebridge_venv, "bin", "python"]),
      Path.join([snakebridge_venv, "Scripts", "python.exe"]),
      Path.join([snakebridge_venv, "Scripts", "python"])
    ]
    |> Enum.find(&File.exists?/1)
  end

if snakebridge_venv do
  config :snakebridge, venv_path: snakebridge_venv
end

if python_executable do
  config :snakepit, python_executable: python_executable
end

# Track current Mix environment for runtime diagnostics
# Use :ml_inference profile for long-running model loading + inference
config :snakepit,
  environment: config_env(),
  timeout_profile: :ml_inference

config :logger,
  level: :warning

# Snakepit is configured in runtime.exs using VLLM.ConfigHelper

import_config "#{config_env()}.exs"
