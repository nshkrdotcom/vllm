# Structured Outputs

Structured outputs allow you to constrain LLM generation to specific formats like JSON, regex patterns, or predefined choices.

## Overview

vLLM supports guided decoding through:
- **JSON Schema** - Output valid JSON matching a schema
- **Regex** - Output matching a regular expression
- **Choice** - Output from a predefined list
- **Grammar** - Output matching a BNF grammar

Note: Guided decoding requires a vLLM build that exposes `GuidedDecodingParams`.
Check `VLLM.guided_decoding_supported?/0` before using these APIs.

## Guided Decoding Parameters

```elixir
# Create guided decoding params
guided = VLLM.guided_decoding_params!(
  json: schema,
  # OR regex: pattern,
  # OR choice: options,
  # OR grammar: bnf
)
```

## JSON Schema

Constrain output to valid JSON matching a schema:

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("meta-llama/Llama-2-7b-hf")

  schema = ~s({
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "age": {"type": "integer", "minimum": 0},
      "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"]
  })

  guided = VLLM.guided_decoding_params!(json: schema)
  params = VLLM.sampling_params!(max_tokens: 100)

  outputs = VLLM.generate!(llm,
    "Extract user info from: John is 25 years old, email john@example.com",
    sampling_params: params,
    guided_decoding_params: guided
  )
end)
```

## Regex Constraints

Match output to a regular expression:

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("meta-llama/Llama-2-7b-hf")

  # Phone number pattern
  guided = VLLM.guided_decoding_params!(regex: "[0-9]{3}-[0-9]{3}-[0-9]{4}")

  outputs = VLLM.generate!(llm,
    "Generate a US phone number:",
    guided_decoding_params: guided
  )
  # Output: "555-123-4567"
end)
```

Common regex patterns:

```elixir
# Date (YYYY-MM-DD)
VLLM.guided_decoding_params!(regex: "[0-9]{4}-[0-9]{2}-[0-9]{2}")

# Email
VLLM.guided_decoding_params!(regex: "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")

# Integer
VLLM.guided_decoding_params!(regex: "-?[0-9]+")

# Float
VLLM.guided_decoding_params!(regex: "-?[0-9]+\\.?[0-9]*")
```

## Choice Constraints

Limit output to specific options:

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("meta-llama/Llama-2-7b-hf")

  # Sentiment classification
  guided = VLLM.guided_decoding_params!(choice: ["positive", "negative", "neutral"])

  outputs = VLLM.generate!(llm,
    "Classify the sentiment of 'I love this product!': ",
    guided_decoding_params: guided
  )
  # Output: "positive"
end)
```

## Grammar Constraints

Use BNF grammar for complex formats:

```elixir
VLLM.run(fn ->
  llm = VLLM.llm!("meta-llama/Llama-2-7b-hf")

  # Simple arithmetic grammar
  grammar = """
  ?start: expr
  ?expr: term (("+" | "-") term)*
  ?term: factor (("*" | "/") factor)*
  ?factor: NUMBER | "(" expr ")"
  NUMBER: /[0-9]+/
  """

  guided = VLLM.guided_decoding_params!(grammar: grammar)

  outputs = VLLM.generate!(llm,
    "Write a math expression:",
    guided_decoding_params: guided
  )
end)
```

## Backend Selection

vLLM supports different guided decoding backends:

```elixir
llm = VLLM.llm!("meta-llama/Llama-2-7b-hf",
  guided_decoding_backend: "outlines"  # or "lm-format-enforcer"
)
```

## Use Cases

### Structured Data Extraction

```elixir
schema = ~s({
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "type": {"type": "string", "enum": ["person", "organization", "location"]}
        }
      }
    }
  }
})

guided = VLLM.guided_decoding_params!(json: schema)
```

### Classification

```elixir
categories = ["technology", "sports", "politics", "entertainment", "science"]
guided = VLLM.guided_decoding_params!(choice: categories)
```

### Form Validation

```elixir
# ZIP code
guided = VLLM.guided_decoding_params!(regex: "[0-9]{5}(-[0-9]{4})?")

# Social Security Number
guided = VLLM.guided_decoding_params!(regex: "[0-9]{3}-[0-9]{2}-[0-9]{4}")
```

## Tips

1. **Simpler is better**: Use choice for finite options, regex for patterns
2. **Test schemas**: Validate JSON schemas work before deployment
3. **Performance**: Constrained decoding adds overhead; balance quality vs speed
4. **Model compatibility**: Ensure model can follow constraints (instruction-tuned helps)

## Limitations

1. **Complex grammars**: Very complex grammars may slow generation
2. **Schema validation**: Runtime validation, not compile-time
3. **Model capabilities**: Model must understand the task to produce valid output
4. **Token boundaries**: Regex/grammar constraints work at token level
