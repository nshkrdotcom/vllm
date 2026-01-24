#!/usr/bin/env python3
"""
Basic vLLM Text Generation Example (Python)

This is a direct Python equivalent of the Elixir basic.exs example
to test if vLLM works correctly on this system.
"""

from vllm import LLM, SamplingParams

def main():
    print("=== Basic vLLM Text Generation (Python) ===\n")

    print("Note: vLLM requires a CUDA-capable GPU.")
    print("If you see CUDA errors, ensure you have a compatible NVIDIA GPU.\n")

    # Create an LLM instance with a small model
    print("Loading model: facebook/opt-125m")
    llm = LLM(
        model="facebook/opt-125m",
        dtype="auto",
        gpu_memory_utilization=0.8
    )

    # Define prompts to complete
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Machine learning is"
    ]

    print(f"\nGenerating completions for {len(prompts)} prompts...\n")

    # Generate completions
    sampling_params = SamplingParams(max_tokens=50)
    outputs = llm.generate(prompts, sampling_params)

    # Process and display results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text

        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("---")

    print("\nBasic generation complete!")

if __name__ == "__main__":
    main()
