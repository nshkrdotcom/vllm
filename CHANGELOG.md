# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-06

### Changed

- Upgraded SnakeBridge from 0.15.1 to 0.16.0 (Snakepit 0.12.0 to 0.13.0)
- Regenerated all vLLM wrappers with SnakeBridge v0.16.0

## [0.2.1] - 2026-01-25

### Changed

- Updated to SnakeBridge 0.15.1
- Regenerated docs with improved enum/member summaries and doctest rendering

## [0.2.0] - 2026-01-25

### Added

- vLLM *public* API surface generation via SnakeBridge (`module_mode: :docs`, driven by committed `priv/snakebridge/vllm.docs.json`)
- Class method guardrail (`max_class_methods`) to prevent extremely large wrappers from inheritance-heavy internal classes
- Coverage reports in `.snakebridge/coverage/` for tracking API binding completeness
- Credo configuration (`.credo.exs`) to exclude generated wrappers from static analysis
- Python test script (`basic_test.py`) for direct vLLM validation
- vLLM v1 multiprocessing configuration option (`:auto`, `:on`, `:off`)
- Auto-set `VLLM_WORKER_MULTIPROC_METHOD=spawn` when v1 multiprocessing is enabled (avoids forking multi-threaded gRPC server)
- Documentation for `mix snakebridge.regen` command with `--clean` option

### Changed

- Updated to SnakeBridge 0.15.0
- Enhanced configuration guide with vLLM v1 multiprocessing documentation
- Updated `direct_api.exs` to be wrapper-only while still demonstrating runtime attribute access for Python refs
- Enhanced `run_all.sh` to compile once and run examples with `--no-compile` (avoids repeated codegen), plus better process cleanup and interrupt handling

### Fixed
- Dialyzer no longer hangs due to an excessively-large generated wrapper surface (docs-manifest generation + class method guardrails keep modules small).
- Examples now create wrapper refs in the same runtime session as the LLM and pass `__runtime__` opts, preventing cross-session reference errors.
- Example docs and scripts now derive `runtime_opts` from the LLM ref consistently for generation, chat, and embeddings calls.

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

[Unreleased]: https://github.com/nshkrdotcom/vllm/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/nshkrdotcom/vllm/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/nshkrdotcom/vllm/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/nshkrdotcom/vllm/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/nshkrdotcom/vllm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nshkrdotcom/vllm/releases/tag/v0.1.0
