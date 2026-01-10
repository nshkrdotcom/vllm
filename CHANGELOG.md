# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-01-10

### Changed

- Embedding models now use `runner: "pooling"` instead of `task: "embed"`
- The `embed/3` function now calls the `embed` method instead of `encode`

### Improved

- Rewrote examples to be runnable with real GPU inference
- Added CLI flag support for examples (--model, --prompt, --adapter, etc.)
- LoRA example auto-downloads a default adapter on first run
- Updated documentation to reflect embedding API changes

### Fixed

- Added generated files to .gitignore (examples/assets/, registry.json)

## [0.1.0] - 2026-01-08

- Initial release

[0.1.1]: https://github.com/nshkrdotcom/vllm/releases/tag/v0.1.1
[0.1.0]: https://github.com/nshkrdotcom/vllm/releases/tag/v0.1.0
