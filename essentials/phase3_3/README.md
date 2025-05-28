# Phase 3.3: Vector Storage and Advanced Retrieval

This module implements a comprehensive vector storage and retrieval system using ChromaDB with support for:

- Persistent storage of vector embeddings
- Advanced retrieval strategies
- Evaluation metrics for retrieval quality

## Components

### 1. ChromaDB Vector Store (`vector_store.py`)

A persistent vector store implementation using ChromaDB with support for:
- Collection management (create, reset, delete)
- Document chunking conversion and storage
- Metadata filtering
- Vector similarity search

```python
from essentials.phase3_3.vector_store import ChromaVectorStore

# Initialize vector store
vector_store = ChromaVectorStore(
    persist_directory="data/chroma_db",
    collection_name="my_collection",
    embedding_dimension=384
)

# Add documents
vector_store.add_documents([
    {
        "id": "doc1",
        "text": "This is a sample document",
        "embedding": [...],  # Embedding vector
        "metadata": {"source": "article", "category": "science"}
    }
])

# Search
results = vector_store.search(
    query_embedding=[...],  # Query embedding vector
    filter_metadata={"category": "science"},
    k=5
)
```

### 2. Advanced Retriever (`retriever.py`)

Implements various retrieval strategies including:
- Semantic search using vector similarity
- Hybrid search combining keyword + vector retrieval
- Filtering by metadata fields
- Maximum Marginal Relevance (MMR) for diversity in results

```python
from essentials.phase3_3.retriever import AdvancedRetriever

# Initialize retriever with vector store and embedding model
retriever = AdvancedRetriever(vector_store, embedding_model)

# Basic semantic search
results = retriever.retrieve("What is quantum computing?", k=5)

# Hybrid search (semantic + keyword)
results = retriever.hybrid_retrieve(
    "quantum entanglement experiments",
    semantic_weight=0.7,
    keyword_weight=0.3,
    k=5
)

# MMR search for diversity
results = retriever.retrieve_with_mmr(
    "machine learning applications",
    lambda_param=0.5,  # Balance between relevance and diversity
    k=5
)
```

### 3. Retrieval Evaluation (`retrieval_evaluation.py`)

Tools for evaluating retrieval quality:
- Precision, recall, and F1 metrics
- Mean Average Precision (MAP) calculation
- Mean Reciprocal Rank (MRR) measurement
- Support for ground truth comparison
- Visualization options for result analysis
- Iterative feedback-based improvement

```python
from essentials.phase3_3.retrieval_evaluation import RetrievalEvaluator

# Initialize evaluator
evaluator = RetrievalEvaluator(retriever, vector_store)

# Evaluate retrieval function on test queries
results = evaluator.evaluate_retrieval(
    queries=["What is AI?", "Explain neural networks"],
    ground_truth=[["doc1", "doc5"], ["doc3", "doc7", "doc9"]],
    retrieval_fn=lambda q: retriever.retrieve(q, k=5),
    k_values=[1, 3, 5],
    run_name="semantic_retrieval"
)

# Compare different retrieval methods
comparison = evaluator.compare_runs(
    ["semantic_retrieval", "hybrid_retrieval", "mmr_retrieval"]
)

# Visualize comparison
evaluator.visualize_comparison(comparison)
```

## Installation

1. Install required dependencies:

```bash
pip install -r essentials/phase3.3/requirements.txt
```

2. For better embedding support, install one of:
   - `sentence-transformers` (recommended)
   - `spacy` with a medium or large model
   - `scikit-learn` (for basic embeddings)

## Usage Example

See `test_vector_store_retrieval.py` for a complete example of how to:
1. Initialize and use ChromaDB vector store
2. Convert and store document chunks
3. Perform various types of semantic search
4. Evaluate retrieval quality

## Integration with Previous Phases

- **Phase 3.1 (Chunking)**: Takes chunked documents from Phase 3.1 and stores them efficiently
- **Phase 3.2 (Embeddings)**: Uses embeddings from Phase 3.2 or can work with any embedding model

## Notes

- ChromaDB provides persistent storage by default
- All retrieval operations support metadata filtering
- The evaluation module helps tune retrieval parameters for optimal performance 