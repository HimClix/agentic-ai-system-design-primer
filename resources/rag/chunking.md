# Chunking Strategies for RAG
> Chunk size is the most impactful parameter in your RAG pipeline. Too small: you lose context. Too large: you dilute relevance. The sweet spot is 512-1024 tokens with semantic boundaries.

## What It Is

Chunking is the process of splitting documents into smaller pieces for embedding and retrieval. Since embedding models have token limits (512-8192 tokens) and retrieval works best with focused text segments, raw documents must be divided into chunks that balance three competing goals:

1. **Semantic coherence**: Each chunk should contain a complete idea
2. **Retrieval precision**: Smaller chunks match specific queries better
3. **Context preservation**: Chunks need enough surrounding context to be useful

The choice of chunking strategy affects every downstream metric: retrieval recall, reranker precision, answer quality, and cost.

## How It Works

### Chunking Strategies Comparison

| Strategy | How It Works | Chunk Size | Overlap | Best For |
|----------|-------------|-----------|---------|----------|
| Fixed-size | Split every N tokens | 256-1024 | 50-100 tokens | Simple pipelines, homogeneous docs |
| Sentence-based | Split on sentence boundaries | 3-10 sentences | 1-2 sentences | Conversational content, articles |
| Semantic | Group sentences by embedding similarity | 200-800 tokens | Dynamic | Mixed-topic documents |
| Recursive | Split by hierarchy: headers > paragraphs > sentences | 512-1024 | 50-100 tokens | Structured documents (Markdown, HTML) |
| Document-level | One chunk per document | Full doc | None | Short documents (emails, tickets) |
| Parent-child | Small chunks for retrieval, return parent section | Child: 128, Parent: 1024 | Implicit | When precision + context both matter |

### Visual: How Each Strategy Splits the Same Text

```
Original Document (2000 tokens):
[====================================================================]
  Section 1: Intro     |  Section 2: Details     |  Section 3: FAQ

Fixed-Size (500 tokens, 50 overlap):
[========]
      [========]
            [========]
                  [========]

Sentence-Based (5 sentences per chunk):
[=====.....][=====.....][=====.....][=====.....]

Semantic (by topic similarity):
[======][============][====]
 Intro      Details     FAQ

Recursive (by structure):
[======]  [============]  [====]
 Intro     Details (split if > max)   FAQ
           [======][======]
           Detail-A  Detail-B

Parent-Child:
Parent: [====================================================================]
Children: [==][==][==][==][==][==][==][==][==][==]
          (Retrieve children, return parent for context)
```

## Production Implementation

### Fixed-Size Chunking (Baseline)

```python
"""Fixed-size chunking with token-based splitting."""
from typing import Iterator
import tiktoken


def chunk_fixed_size(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    model: str = "text-embedding-3-small",
) -> list[dict]:
    """Split text into fixed-size token chunks with overlap.
    
    Args:
        text: Input document text
        chunk_size: Target chunk size in tokens
        overlap: Number of overlapping tokens between consecutive chunks
        model: Tokenizer model name
    
    Returns:
        List of chunks with text and metadata
    """
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        
        chunks.append({
            "text": chunk_text,
            "token_count": len(chunk_tokens),
            "chunk_index": chunk_index,
            "start_token": start,
            "end_token": end,
        })
        
        # Move forward by (chunk_size - overlap)
        start += chunk_size - overlap
        chunk_index += 1
    
    return chunks
```

### Semantic Chunking

```python
"""Semantic chunking: group sentences by embedding similarity."""
import numpy as np
from sentence_transformers import SentenceTransformer
import re


def chunk_semantic(
    text: str,
    max_chunk_tokens: int = 800,
    similarity_threshold: float = 0.75,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """Split text into semantically coherent chunks.
    
    Groups consecutive sentences that are semantically similar.
    Starts a new chunk when similarity drops below threshold.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return [{"text": text, "chunk_index": 0}]
    
    # Embed all sentences
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(sentences)
    
    # Group by similarity
    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = embeddings[0]
    
    for i in range(1, len(sentences)):
        # Cosine similarity between current chunk centroid and next sentence
        similarity = np.dot(current_embedding, embeddings[i]) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(embeddings[i])
        )
        
        if similarity >= similarity_threshold:
            # Same topic: add to current chunk
            current_chunk.append(sentences[i])
            # Update centroid (running average)
            current_embedding = np.mean(
                [current_embedding, embeddings[i]], axis=0
            )
        else:
            # Topic shift: start new chunk
            chunks.append({
                "text": " ".join(current_chunk),
                "chunk_index": len(chunks),
                "sentence_count": len(current_chunk),
            })
            current_chunk = [sentences[i]]
            current_embedding = embeddings[i]
    
    # Final chunk
    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "chunk_index": len(chunks),
            "sentence_count": len(current_chunk),
        })
    
    return chunks
```

### Recursive Chunking (Best for Structured Documents)

```python
"""Recursive chunking: respect document structure."""
import re
from typing import Optional
import tiktoken


# Separators in priority order (try the most structural first)
MARKDOWN_SEPARATORS = [
    "\n## ",       # H2 headers
    "\n### ",      # H3 headers  
    "\n#### ",     # H4 headers
    "\n\n",        # Paragraph breaks
    "\n",          # Line breaks
    ". ",          # Sentences
    " ",           # Words (last resort)
]


def chunk_recursive(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    separators: list[str] = None,
) -> list[dict]:
    """Recursively split text using hierarchical separators.
    
    Tries to split on the most structural boundary first
    (headers, then paragraphs, then sentences, then words).
    """
    if separators is None:
        separators = MARKDOWN_SEPARATORS
    
    encoder = tiktoken.encoding_for_model("text-embedding-3-small")
    
    def _token_count(text: str) -> int:
        return len(encoder.encode(text))
    
    def _split(text: str, seps: list[str]) -> list[str]:
        if _token_count(text) <= chunk_size:
            return [text]
        
        if not seps:
            # No separators left; force split by tokens
            tokens = encoder.encode(text)
            result = []
            for i in range(0, len(tokens), chunk_size - overlap):
                chunk_tokens = tokens[i:i + chunk_size]
                result.append(encoder.decode(chunk_tokens))
            return result
        
        sep = seps[0]
        remaining_seps = seps[1:]
        
        parts = text.split(sep)
        
        chunks = []
        current = ""
        
        for part in parts:
            candidate = current + sep + part if current else part
            
            if _token_count(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If this single part is too large, recurse with finer separators
                if _token_count(part) > chunk_size:
                    sub_chunks = _split(part, remaining_seps)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part
        
        if current:
            chunks.append(current)
        
        return chunks
    
    raw_chunks = _split(text, separators)
    
    return [
        {
            "text": chunk.strip(),
            "chunk_index": i,
            "token_count": _token_count(chunk.strip()),
        }
        for i, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]
```

### Parent-Child (Contextual Retrieval Pattern)

```python
"""Parent-child chunking: retrieve small, return large."""


def chunk_parent_child(
    text: str,
    parent_size: int = 1024,
    child_size: int = 128,
    child_overlap: int = 20,
) -> list[dict]:
    """Create parent-child chunk pairs.
    
    Small children are used for precise retrieval.
    When a child is retrieved, the parent is returned for context.
    """
    encoder = tiktoken.encoding_for_model("text-embedding-3-small")
    tokens = encoder.encode(text)
    
    results = []
    parent_idx = 0
    
    # Create parent chunks
    for p_start in range(0, len(tokens), parent_size):
        p_end = min(p_start + parent_size, len(tokens))
        parent_tokens = tokens[p_start:p_end]
        parent_text = encoder.decode(parent_tokens)
        parent_id = f"parent_{parent_idx}"
        
        # Create child chunks within this parent
        child_idx = 0
        for c_start in range(0, len(parent_tokens), child_size - child_overlap):
            c_end = min(c_start + child_size, len(parent_tokens))
            child_tokens = parent_tokens[c_start:c_end]
            child_text = encoder.decode(child_tokens)
            
            results.append({
                "child_text": child_text,        # Embed this
                "parent_text": parent_text,      # Return this to LLM
                "child_id": f"{parent_id}_child_{child_idx}",
                "parent_id": parent_id,
                "child_token_count": len(child_tokens),
                "parent_token_count": len(parent_tokens),
            })
            child_idx += 1
        
        parent_idx += 1
    
    return results
```

## Decision Tree: Choosing a Chunking Strategy

```
What type of documents are you indexing?
  |
  |-- Structured (Markdown, HTML, code)?
  |     --> Recursive chunking (respects headers, sections)
  |
  |-- Unstructured prose (articles, reports)?
  |     --> Semantic chunking (groups by topic similarity)
  |
  |-- Short documents (emails, tickets, chat)?
  |     --> Document-level (one chunk per document)
  |
  |-- Mixed content (docs with tables, code, prose)?
  |     --> Recursive chunking + special handling for tables/code
  |
  |-- Need both precise retrieval AND full context?
  |     --> Parent-child (small retrieval units, large context units)
  |
  |-- Just getting started / prototyping?
        --> Fixed-size 512 tokens, 50 overlap (simple baseline)
```

## When NOT to Chunk

- **Short documents** (<256 tokens): Embed the full document as one chunk
- **Structured data**: Tables, CSVs, JSON -- use structured querying (SQL/API), not RAG
- **Code files**: Use AST-aware splitting, not text chunking
- **Conversation logs**: Chunk by conversation turn, not by token count

## Tradeoffs

| Parameter | Smaller Chunks (128-256) | Medium Chunks (512-1024) | Larger Chunks (1024-2048) |
|-----------|------------------------|-------------------------|--------------------------|
| Retrieval precision | High (specific matches) | Balanced | Low (diluted relevance) |
| Context preservation | Low (loses surrounding info) | Balanced | High (complete sections) |
| Number of chunks | Many (more storage, slower search) | Moderate | Few |
| Embedding cost | Higher (more embeddings) | Moderate | Lower |
| Best for | Fact-based Q&A | General knowledge | Summarization, analysis |

## Real-World Examples

### Chunk Size Impact on Retrieval Quality
```
Dataset: 10,000 technical documentation pages
Query set: 500 real user questions
Metric: Recall@5 (correct doc in top 5 results)

Chunk size 128:  Recall@5 = 78%  (too fragmented, loses context)
Chunk size 256:  Recall@5 = 83%
Chunk size 512:  Recall@5 = 89%  <-- sweet spot
Chunk size 1024: Recall@5 = 86%  (starts to dilute)
Chunk size 2048: Recall@5 = 79%  (too much noise per chunk)
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| Split mid-sentence | Fixed-size splitting ignores boundaries | Incoherent chunks, poor retrieval | Use sentence-aware splitting |
| Lost context | Chunks too small | Retrieved chunk is meaningless alone | Increase chunk size or use parent-child |
| Diluted relevance | Chunks too large | Irrelevant text included in retrieval | Decrease chunk size, use reranker |
| Table destruction | Splitting through a table | Table data becomes gibberish | Detect tables, keep them as single chunks |
| Duplicate content | Large overlap between chunks | Wasted storage, redundant retrieval | Keep overlap to 10-15% of chunk size |
| Header orphaning | Section header in one chunk, content in next | Header chunk is useless | Prepend header to each child chunk |

## Source(s) and Further Reading

- RAG_Techniques repository (GitHub, 26.2k stars) -- chunking strategy catalog
- Anthropic, "Contextual Retrieval" (2024) -- parent-child chunking + contextual embeddings
- LlamaIndex, "Chunking Strategies" -- recursive and semantic chunking implementations
- Greg Kamradt, "5 Levels of Text Splitting" (YouTube) -- visual guide to chunking strategies
- Pinecone, "Chunking Strategies for LLM Applications" -- production recommendations
