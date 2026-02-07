defmodule VLLM.MixProject do
  use Mix.Project

  @version "0.3.0"
  @source_url "https://github.com/nshkrdotcom/vllm"

  def project do
    [
      app: :vllm,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      python_deps: python_deps(),
      elixirc_paths: elixirc_paths(Mix.env()),
      preferred_cli_env: [
        dialyzer: :dialyzer,
        "dialyzer.clean": :dialyzer,
        "dialyzer.plt": :dialyzer
      ],
      # Add snakebridge compiler for Python introspection and auto-install
      compilers: compilers(Mix.env()),

      # Dialyzer
      dialyzer: [
        plt_add_apps: [:mix],
        plt_file: {:no_warn, "priv/plts/dialyzer.plt"}
      ],

      # Package info
      name: "VLLM",
      description: description(),
      source_url: @source_url,
      homepage_url: @source_url,
      docs: docs(),
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      # SnakeBridge - Python bridge for vLLM (local dev)
      {:snakebridge, "~> 0.16.0"},

      # JSON encoding
      {:jason, "~> 1.4"},

      # Dev/test tools
      {:ex_doc, "~> 0.40", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test, :dialyzer], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}
    ]
  end

  defp python_deps do
    [
      # Generate vLLM's *documented* public surface from a committed docs manifest.
      {:vllm, "0.14.0",
       generate: :all,
       module_mode: :docs,
       docs_manifest: "priv/snakebridge/vllm.docs.json",
       docs_profile: :full,
       max_class_methods: 500}
    ]
  end

  defp elixirc_paths(:test), do: ["lib/vllm", "lib/snakebridge_generated", "test/support"]
  defp elixirc_paths(_), do: ["lib/vllm", "lib/snakebridge_generated"]

  defp compilers(:dialyzer), do: Mix.compilers()
  defp compilers(_), do: [:snakebridge] ++ Mix.compilers()

  defp description do
    """
    vLLM for Elixir via SnakeBridge - Easy, fast, and cheap LLM serving for everyone.
    High-throughput LLM inference with PagedAttention, continuous batching, and OpenAI-compatible API.
    """
  end

  defp docs do
    [
      main: "readme",
      name: "VLLM",
      source_ref: "v#{@version}",
      source_url: @source_url,
      homepage_url: @source_url,
      assets: %{"assets" => "assets"},
      logo: "assets/vllm.svg",
      extras: [
        "README.md",
        {"examples/README.md", [filename: "examples"]},
        "docs/guides/quickstart.md",
        "docs/guides/offline_inference.md",
        "docs/guides/online_serving.md",
        "docs/guides/sampling_params.md",
        "docs/guides/configuration.md",
        "docs/guides/supported_models.md",
        "docs/guides/quantization.md",
        "docs/guides/multimodal.md",
        "docs/guides/lora.md",
        "docs/guides/structured_outputs.md",
        "CHANGELOG.md",
        "LICENSE"
      ],
      groups_for_extras: [
        Guides: ["README.md", "docs/guides/quickstart.md"],
        Features: [
          "docs/guides/offline_inference.md",
          "docs/guides/online_serving.md",
          "docs/guides/sampling_params.md",
          "docs/guides/configuration.md",
          "docs/guides/multimodal.md",
          "docs/guides/lora.md",
          "docs/guides/structured_outputs.md"
        ],
        Reference: [
          "docs/guides/supported_models.md",
          "docs/guides/quantization.md"
        ],
        Examples: ["examples/README.md"],
        "Release Notes": ["CHANGELOG.md"]
      ],
      groups_for_modules: [
        "Core API": [VLLM],
        "Sampling & Parameters": [VLLM.SamplingParams],
        "Engine & Server": [VLLM.Engine, VLLM.Server],
        Configuration: [VLLM.Config]
      ],
      before_closing_head_tag: fn
        :html ->
          """
          <script defer src="https://cdn.jsdelivr.net/npm/mermaid@10.2.3/dist/mermaid.min.js"></script>
          <script>
            let initialized = false;

            window.addEventListener("exdoc:loaded", () => {
              if (!initialized) {
                mermaid.initialize({
                  startOnLoad: false,
                  theme: document.body.className.includes("dark") ? "dark" : "default"
                });
                initialized = true;
              }

              let id = 0;
              for (const codeEl of document.querySelectorAll("pre code.mermaid")) {
                const preEl = codeEl.parentElement;
                const graphDefinition = codeEl.textContent;
                const graphEl = document.createElement("div");
                const graphId = "mermaid-graph-" + id++;
                mermaid.render(graphId, graphDefinition).then(({svg, bindFunctions}) => {
                  graphEl.innerHTML = svg;
                  bindFunctions?.(graphEl);
                  preEl.insertAdjacentElement("afterend", graphEl);
                  preEl.remove();
                });
              }
            });
          </script>
          """

        _ ->
          ""
      end
    ]
  end

  defp package do
    [
      name: "vllm",
      description: description(),
      files: ~w(lib mix.exs README.md CHANGELOG.md LICENSE),
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "vLLM Python" => "https://github.com/vllm-project/vllm"
      },
      maintainers: ["nshkrdotcom"],
      exclude_patterns: [
        "priv/plts",
        ".DS_Store"
      ]
    ]
  end
end
