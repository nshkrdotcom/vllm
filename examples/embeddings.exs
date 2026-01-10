# Embeddings Example
#
# This example demonstrates using vLLM for embedding generation:
# - Loading embedding models
# - Generating text embeddings
# - Batch embedding
#
# Run: mix run examples/embeddings.exs
#
# Note: Requires an embedding-capable model

IO.puts("=== Embeddings Example ===\n")

IO.puts("--- Supported Embedding Models ---")

IO.puts("""
vLLM supports various embedding/pooling models:

| Model                          | Dimensions | Use Case          |
|--------------------------------|------------|-------------------|
| intfloat/e5-mistral-7b-instruct| 4096       | General purpose   |
| BAAI/bge-large-en-v1.5         | 1024       | English text      |
| sentence-transformers/all-*    | Various    | Sentence similarity|
""")

IO.puts("--- Example: Loading Embedding Model ---")

IO.puts("""
# Load an embedding model
llm = VLLM.llm!("intfloat/e5-mistral-7b-instruct",
  task: "embed",
  dtype: "float16"
)

# Generate embeddings
texts = ["Hello, world!", "How are you?"]
outputs = VLLM.embed!(llm, texts)
""")

IO.puts("--- Example: Batch Embedding ---")

IO.puts("""
# Efficient batch embedding
documents = [
  "The quick brown fox jumps over the lazy dog.",
  "Machine learning is transforming industries.",
  "Elixir is a functional programming language.",
  "vLLM provides high-throughput inference."
]

embeddings = VLLM.embed!(llm, documents)

# Process embeddings
Enum.each(embeddings, fn output ->
  embedding = VLLM.attr!(output, "outputs")
  # embedding is a list of floats (vector)
end)
""")

IO.puts("--- Embedding Use Cases ---")

IO.puts("""
Common use cases for embeddings:

1. Semantic Search
   - Convert queries and documents to embeddings
   - Find similar documents using cosine similarity

2. Clustering
   - Group similar texts together
   - Topic modeling

3. RAG (Retrieval-Augmented Generation)
   - Embed knowledge base documents
   - Retrieve relevant context for LLM prompts

4. Classification
   - Use embeddings as features for classifiers
""")

IO.puts("--- Pooling Parameters ---")

IO.puts("""
# Create pooling parameters
params = VLLM.pooling_params!(additional_data: %{})

# Use with embed
outputs = VLLM.embed!(llm, texts, pooling_params: params)
""")

# Note: Actual embedding demo would require compatible model
IO.puts("\n--- Note ---")
IO.puts("Actual embedding generation requires an embedding-capable model.")
IO.puts("Install models like 'intfloat/e5-mistral-7b-instruct' for full functionality.")

IO.puts("\nEmbeddings example complete!")
