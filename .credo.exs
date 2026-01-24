%{
  configs: [
    %{
      name: "default",
      files: %{
        included: ["lib/", "test/", "config/"],
        excluded: [
          "lib/snakebridge_generated/",
          ~r"/_build/",
          ~r"/deps/",
          ~r"/node_modules/"
        ]
      },
      strict: true,
      # Generated wrapper modules can be huge; keep Credo responsive for real code.
      parse_timeout: 10_000,
      color: true
    }
  ]
}
