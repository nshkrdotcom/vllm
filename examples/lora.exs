# LoRA Adapters Example
#
# This example demonstrates using LoRA adapters with vLLM:
# - Loading a base model with LoRA support
# - Creating a LoRA request
# - Generating with LoRA weights
#
# IMPORTANT: vLLM requires a CUDA-capable GPU.
#
# Run: mix run examples/lora.exs
#
# Default adapter is downloaded automatically on first run.

defmodule VLLM.Examples.LoraHelper do
  @default_repo "edbeeching/opt-125m-lora"
  @default_base_model "facebook/opt-125m"
  @adapter_files ["adapter_model.safetensors", "adapter_model.bin"]

  def default_repo, do: @default_repo
  def default_base_model, do: @default_base_model

  def default_adapter_dir do
    repo_dir = @default_repo |> String.replace("/", "_")
    Path.expand(Path.join([__DIR__, "assets", "lora", repo_dir]))
  end

  def ensure_adapter!(dir, repo \\ @default_repo) do
    File.mkdir_p!(dir)

    unless File.exists?(Path.join(dir, "adapter_config.json")) do
      download!(repo, "adapter_config.json", dir)
    end

    if Enum.any?(@adapter_files, &File.exists?(Path.join(dir, &1))) do
      :ok
    else
      downloaded =
        Enum.find_value(@adapter_files, fn filename ->
          case download(repo, filename, dir) do
            :ok -> filename
            {:error, _} -> nil
          end
        end)

      if is_nil(downloaded) do
        IO.puts("Failed to download LoRA adapter weights for #{repo}.")
        System.halt(1)
      end
    end
  end

  def read_adapter_config(dir) do
    config_path = Path.join(dir, "adapter_config.json")

    if File.exists?(config_path) do
      case Jason.decode(File.read!(config_path)) do
        {:ok, config} -> config
        {:error, _} -> %{}
      end
    else
      %{}
    end
  end

  defp download!(repo, filename, dir) do
    case download(repo, filename, dir) do
      :ok ->
        :ok

      {:error, reason} ->
        IO.puts("Failed to download #{filename} from #{repo}: #{inspect(reason)}")
        System.halt(1)
    end
  end

  defp download(repo, filename, dir) do
    url = "https://huggingface.co/#{repo}/resolve/main/#{filename}"
    path = Path.join(dir, filename)

    IO.puts("Downloading #{filename} from #{repo}...")

    case download_via_cli(url, path) do
      :ok -> :ok
      {:error, _} -> download_via_httpc(url, path)
    end
  end

  defp download_via_cli(url, path) do
    cond do
      exe = System.find_executable("curl") ->
        case System.cmd(exe, ["-fL", "-o", path, url], stderr_to_stdout: true) do
          {_, 0} -> :ok
          {output, _} -> {:error, output}
        end

      exe = System.find_executable("wget") ->
        case System.cmd(exe, ["-O", path, url], stderr_to_stdout: true) do
          {_, 0} -> :ok
          {output, _} -> {:error, output}
        end

      true ->
        {:error, :no_downloader}
    end
  end

  defp download_via_httpc(url, path) do
    :inets.start()
    :ssl.start()

    request = {String.to_charlist(url), []}

    case :httpc.request(:get, request, [], body_format: :binary) do
      {:ok, {{_, 200, _}, _headers, body}} ->
        File.write!(path, body)
        :ok

      {:ok, {{_, status, _}, _headers, body}} ->
        {:error, {:http_error, status, body}}

      {:error, reason} ->
        {:error, reason}
    end
  end
end

IO.puts("=== LoRA Adapters Example ===\n")
IO.puts("Note: vLLM requires a CUDA-capable GPU.\n")

{opts, _, _} =
  OptionParser.parse(System.argv(),
    switches: [
      model: :string,
      adapter: :string,
      prompt: :string,
      name: :string,
      rank: :integer
    ]
  )

default_adapter_path = VLLM.Examples.LoraHelper.default_adapter_dir()
adapter_path = opts[:adapter] || default_adapter_path

if adapter_path == "" do
  IO.puts("LoRA adapter path is required for this example.")
  IO.puts("Pass --adapter /path/to/adapter or place one at #{default_adapter_path}")
  System.halt(1)
end

adapter_dir =
  if File.dir?(adapter_path) do
    adapter_path
  else
    Path.dirname(adapter_path)
  end

if adapter_path == default_adapter_path do
  VLLM.Examples.LoraHelper.ensure_adapter!(
    adapter_dir,
    VLLM.Examples.LoraHelper.default_repo()
  )
else
  unless File.exists?(adapter_dir) do
    IO.puts("LoRA adapter path not found: #{adapter_dir}")
    IO.puts("Pass --adapter /path/to/adapter to use a local adapter.")
    System.halt(1)
  end

  required = [
    Path.join(adapter_dir, "adapter_config.json"),
    Path.join(adapter_dir, "adapter_model.bin"),
    Path.join(adapter_dir, "adapter_model.safetensors")
  ]

  unless File.exists?(Enum.at(required, 0)) and
           (File.exists?(Enum.at(required, 1)) or File.exists?(Enum.at(required, 2))) do
    IO.puts("LoRA adapter files not found in: #{adapter_dir}")
    IO.puts("Expected adapter_config.json and adapter_model.bin/safetensors.")
    System.halt(1)
  end
end

adapter_config = VLLM.Examples.LoraHelper.read_adapter_config(adapter_dir)

model =
  opts[:model] || adapter_config["base_model_name_or_path"] ||
    VLLM.Examples.LoraHelper.default_base_model()

if is_nil(model) or model == "" do
  IO.puts("Base model is required for this example.")
  IO.puts("Pass --model model-name or include base_model_name_or_path in adapter_config.json.")
  System.halt(1)
end

lora_name = opts[:name] || "adapter"
lora_prompt = opts[:prompt] || "Write a short SQL query to list users."

lora_rank = opts[:rank] || adapter_config["r"] || 64

IO.puts("Loading base model: #{model}")
IO.puts("Using LoRA adapter: #{adapter_dir}")

VLLM.run(fn ->
  llm =
    VLLM.llm!(model,
      enable_lora: true,
      max_loras: 1,
      max_lora_rank: lora_rank,
      gpu_memory_utilization: 0.8
    )

  lora = VLLM.lora_request!(lora_name, 1, adapter_dir)
  params = VLLM.sampling_params!(temperature: 0.7, max_tokens: 80)

  outputs = VLLM.generate!(llm, lora_prompt, sampling_params: params, lora_request: lora)

  output = Enum.at(outputs, 0)
  completion = VLLM.attr!(output, "outputs") |> Enum.at(0)

  IO.puts("\nPrompt: #{lora_prompt}")
  IO.puts("Response: #{VLLM.attr!(completion, "text")}")

  IO.puts("\nLoRA example complete!")
end)
