# Phase 3.4: Context Building for RAG

This phase implements the context building components for the LangChain-GPT RAG pipeline, focusing on creating high-quality context from retrieved chunks and generating appropriate prompts for the LLM.

## Components

### Context Builder (`context_builder.py`)

The `ContextBuilder` class is responsible for assembling retrieved chunks into a prompt-ready context block. Key features include:

- Sorting chunks by relevance score
- Managing context window token limits
- Including metadata with chunks (source, section, etc.)
- Deduplicating similar or repeated content
- Promoting information diversity across sources
- Implementing coherence checks to prevent topic shifts
- Tracking citations alongside chunk content

```python
from essentials.phase3_4 import ContextBuilder

# Initialize the context builder
builder = ContextBuilder(
    max_tokens=2048,             # Maximum tokens allowed in the context
    include_metadata=True,       # Include metadata with chunks
    deduplicate=True,            # Remove similar chunks
    diversify=True,              # Promote diversity across sources
    coherence_check=True         # Check for topic coherence
)

# Build context from retrieved chunks
context_result = builder.build_context(
    retrieved_chunks=[
        {
            "id": "chunk1",
            "text": "Content of the chunk...",
            "metadata": {"source": "doc1", "section": "intro"},
            "score": 0.95
        },
        # More chunks...
    ]
)

# Access the built context
context_text = context_result["context"]
chunks_used = context_result["chunks_used"]
tokens_used = context_result["tokens_used"]
citations = context_result["citations"]
```

The `context_builder.py` also includes a utility function `trim_text_to_token_limit()` for truncating text to fit within a token budget.

### Deduplication Utils (`deduplication_utils.py`)

This module provides utilities for detecting and handling duplicate or similar content:

- `jaccard_similarity()`: Calculate text similarity using Jaccard index
- `contains_substring()`: Check if one text substantially contains another
- `cosine_similarity()`: Calculate similarity between embedding vectors
- `deduplicate_chunks()`: Remove similar chunks from a collection
- `diversify_chunks()`: Ensure diversity by limiting chunks per source
- `find_duplicates_in_text()`: Find repeated phrases in text
- `remove_duplicated_sentences()`: Remove duplicate sentences

```python
from essentials.phase3_4 import deduplicate_chunks, diversify_chunks

# Remove similar chunks
unique_chunks = deduplicate_chunks(chunks, similarity_threshold=0.7)

# Ensure diversity across sources
diverse_chunks = diversify_chunks(chunks, max_per_source=2)
```

### Prompt Templates (`prompt_templates.py`)

This module provides templates for generating prompts for different query types in a scientific RAG system:

- `PromptTemplate`: Base class for creating prompts with context and instructions
- `PromptTemplateLibrary`: Collection of templates for different query types
- `QueryType`: Enum of query types (General, Methodology, Results, etc.)

The system includes specialized templates for different query types with appropriate system instructions, formatting, and few-shot examples:

- General questions
- Methodology questions
- Results analysis
- Comparison queries
- Definition/explanation queries
- Literature review questions
- Synthesis across sources

```python
from essentials.phase3_4 import PromptTemplateLibrary, QueryType

# Initialize the library
template_library = PromptTemplateLibrary()

# Auto-detect query type and create appropriate prompt
prompt_data = template_library.create_prompt(
    query="What method was used to analyze the data?",
    context="Context text with retrieved information...",
)

# Access the prompt components
system_message = prompt_data["system_message"]
prompt_text = prompt_data["prompt"]

# Or specify a query type directly
specific_prompt = template_library.create_prompt(
    query="Compare method A and method B",
    context="Context about methods...",
    query_type=QueryType.COMPARISON
)
```

## Integration with Previous Phases

The context building system is fully compatible with:

- **Phase 3.1**: Works directly with `Chunk` objects from the document chunking system
- **Phase 3.3**: Processes chunks retrieved from the vector store system

## Example Usage

Here's a complete example showing how to use these components in a RAG pipeline:

```python
from essentials.phase3_1 import Chunk
from essentials.phase3_3 import ChromaVectorStore, AdvancedRetriever
from essentials.phase3_4 import ContextBuilder, PromptTemplateLibrary

# 1. Retrieve chunks from vector store
vector_store = ChromaVectorStore(...)
retriever = AdvancedRetriever(vector_store)
retrieved_chunks = retriever.retrieve("What method was used in the experiment?")

# 2. Build context from retrieved chunks
builder = ContextBuilder(max_tokens=2048)
context_result = builder.build_context(retrieved_chunks)

# 3. Create a prompt with the appropriate template
template_library = PromptTemplateLibrary()
prompt_data = template_library.create_prompt(
    query="What method was used in the experiment?",
    context=context_result["context"]
)

# 4. Use the prompt with an LLM
system_message = prompt_data["system_message"]
prompt_text = prompt_data["prompt"]

# Send to LLM (implementation depends on your LLM API)
# llm_response = call_llm_api(system_message, prompt_text)
```

## Testing

The module includes comprehensive tests in `test_context_building.py` that verify:
- Context building functionality
- Deduplication and diversification
- Token limit enforcement
- Prompt template generation
- Query type detection

Run the tests with:
```
python -m essentials.phase3_4.test_context_building
``` 