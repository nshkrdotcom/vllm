# Multimodal Models

vLLM supports multimodal models that can process images, audio, and video alongside text.

## Supported Model Types

- Vision-Language Models (VLMs) - Images + Text
- Audio Models - Audio + Text
- Video Models - Video + Text

## Vision-Language Models

### LLaVA

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("llava-hf/llava-1.5-7b-hf",
    max_model_len: 4096
  )

  # Note: Image input requires specific formatting
  # depending on the model's expected input format
end)
```

### Qwen-VL

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("Qwen/Qwen2-VL-7B-Instruct")
end)
```

## Image Input Formats

Multimodal inputs typically include:

1. **Image URLs** - Direct links to images
2. **Base64 encoded images** - Inline image data
3. **Local file paths** - Path to image files

Example message format for vision models:

```elixir
messages = [[
  %{
    "role" => "user",
    "content" => [
      %{"type" => "text", "text" => "What's in this image?"},
      %{"type" => "image_url", "image_url" => %{"url" => "https://example.com/image.jpg"}}
    ]
  }
]]
```

## Model-Specific Formats

Different models may expect different input formats. Check the specific model's documentation on HuggingFace for details.

### LLaVA Format

```elixir
# LLaVA uses special tokens for images
prompt = "<image>\\nUSER: What's in this image?\\nASSISTANT:"
```

### Qwen-VL Format

```elixir
# Qwen-VL uses specific message format
messages = [[
  %{
    "role" => "user",
    "content" => [
      %{"type" => "image", "image" => "path/to/image.jpg"},
      %{"type" => "text", "text" => "Describe this image."}
    ]
  }
]]
```

## Configuration for Multimodal

```elixir
llm = VLLM.llm!("llava-hf/llava-1.5-7b-hf",
  # Limit number of images per prompt
  limit_mm_per_prompt: %{"image" => 4},

  # Maximum model length (images use many tokens)
  max_model_len: 8192,

  # Trust remote code for custom processors
  trust_remote_code: true
)
```

## Memory Considerations

Multimodal models are typically larger due to:
- Vision encoder weights
- Cross-modal projection layers
- Higher sequence lengths for images

Consider:
- Using quantization for larger models
- Reducing `max_model_len` if not processing many images
- Using tensor parallelism for very large models

## Limitations

1. **Image processing overhead** - First inference may be slower due to image encoding
2. **Token consumption** - Images consume many tokens (hundreds to thousands)
3. **Model availability** - Not all multimodal models are supported

## Example: Image Description

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("llava-hf/llava-1.5-7b-hf")

  params = VLLM.sampling_params!(
    temperature: 0.7,
    max_tokens: 200
  )

  # Format depends on specific model
  # Consult model documentation for exact format
  prompt = "<image>\\nDescribe this image in detail.\\nASSISTANT:"

  # Note: Actual image data needs to be provided via
  # model-specific mechanisms
end)
```

## Future Support

vLLM continues to expand multimodal support. Check the latest vLLM documentation for newly supported models and features.
