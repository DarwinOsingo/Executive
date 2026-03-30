# Kenya AI Executive Roundtable

A sophisticated Retrieval-Augmented Generation (RAG) pipeline for processing and analyzing policy documents across multiple cabinet sectors. This system enables AI-powered agents to retrieve relevant information from structured and unstructured documents to support executive decision-making.

## Overview

The Kenya AI Executive Roundtable implements a multi-agent RAG system designed to process fiscal policy documents, budget statements, economic surveys, and sector-specific reports across seven cabinet ministries. The pipeline combines document ingestion, intelligent chunking, vector embedding, and semantic search with LLM-based inference.

## Project Structure

```
Executive/
├── Finance/                 # Primary pipeline implementation (financial documents)
│   ├── agent.py            # LLM inference wrapper (Groq/Together AI)
│   ├── retriever.py        # Vector search and context retrieval
│   ├── chunk_documents.py  # Document segmentation logic
│   ├── prepare_data.py     # Data preprocessing and validation
│   ├── upsert.py          # Qdrant vector database operations
│   ├── config.yaml        # Pipeline configuration (documents, chunking, indexing)
│   ├── generate_config.py # Configuration generation from templates
│   ├── rubric.py          # Evaluation metrics and quality assurance
│   ├── datatest.py        # Unit tests for pipeline components
│   ├── result.json        # Pipeline execution results
│   ├── upsert_checkpoint.json  # Checkpoint tracking for incremental updates
│   ├── pipeline/          # Modular pipeline components
│   │   ├── extractor.py   # Information extraction from documents
│   │   └── table_processor.py  # Structured table handling
│   ├── data/              # Data management
│   │   ├── raw/           # Source documents (PDFs)
│   │   ├── cache/         # Processed document metadata (JSON)
│   │   ├── chunks/        # Segmented document chunks
│   │   ├── processed/     # Cleaned and validated data
│   │   └── rag/           # RAG pipeline outputs
│   └── chunks/            # Chunk storage and indexing
│
├── Agriculture/           # Sector-specific pipeline (agriculture domain)
│   └── data/
│       └── raw/          # Agricultural policy documents
│
├── AntiCorruption/       # Anti-corruption focused data
│   ├── data/
│   │   ├── raw/          # Source documents
│   │   └── processed/    # Processed data
│   └── processed/        # Finalized outputs
│
├── Education/            # Education sector documents
├── ICT/                  # Information and Communication Technology sector
├── Infrastructure/       # Infrastructure policy documents
│   └── data/
│       └── raw/          # Infrastructure-related documents
│
├── President/            # Presidential level documents
│
├── data/                 # Cross-sector data repository
│
├── datatest.py          # Integration tests for all sectors
├── ruberic.py           # Global evaluation framework
└── Readme.md            # This file
```

## Key Components

### 1. Agent Module (`Finance/agent.py`)
- Wraps LLM inference for cabinet agents
- Integrates Groq API (Llama 3.1 70B) for development
- Scheduled for fine-tuning with Together AI and DeepSeek R1
- Template-based for replication across 7 cabinet ministr

### 2. Retriever Module (`Finance/retriever.py`)
- Vector similarity search using Qdrant
- Semantic context retrieval
- Integration with FastEmbed/SentenceTransformers for embeddings

### 3. Document Processing Pipeline
- **chunk_documents.py**: Intelligent document segmentation
  - Token-based chunking (100-500 tokens)
  - Strategy-specific processing (narrative vs. tables)
  - Deduplication (threshold: 0.85)

- **table_processor.py**: Specialized handling for tabular data
  - Row-level table chunking
  - Structured data preservation
  - Financial table parsing

- **extractor.py**: Information extraction
  - Entity recognition
  - Relationship extraction
  - Domain-specific field extraction

### 4. Configuration Management
- YAML-based pipeline configuration (`config.yaml`)
- Automatic configuration generation via `generate_config.py`
- Document metadata tracking (type, fiscal year, priority, weights)
- Multi-level review system (none, soft, hard)

### 5. Vector Database Operations
- **upsert.py**: Manages Qdrant collection operations
- **upsert_checkpoint.json**: Tracks incremental updates and prevents re-indexing
- Collection name: `kenya_executive_roundtable`

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (venv, conda, or equivalent)
- API keys for external services (Groq, VoyageAI)

### Setup

1. Clone the repository and navigate to the Executive directory:
```bash
cd /home/darwin/PRES/Executive
source /home/darwin/ml-env/bin/activate
```

2. Install dependencies:
```bash
pip install groq voyageai qdrant-client fastembed sentence-transformers
```

3. Configure environment variables:
```bash
export GROQ_API_KEY="your_groq_api_key"
export VOYAGEAI_API_KEY="your_voyage_api_key"
export QDRANT_HOST="localhost"  # or your Qdrant server
export QDRANT_PORT="6333"
```

## Usage

### Basic Agent Usage

```python
from Finance.agent import Agent

# Initialize an agent
prof_kamau = Agent.from_config("finance")

# Query the agent
response = prof_kamau.speak(
    "The Infrastructure CS is proposing a KSh 50B SGR extension. What are the fiscal implications?"
)
print(response)
```

### Document Processing Pipeline

1. **Prepare data**:
```bash
python Finance/prepare_data.py --input-dir Finance/data/raw --output-dir Finance/data/processed
```

2. **Generate configuration**:
```bash
python Finance/generate_config.py --template config.yaml
```

3. **Process and chunk documents**:
```bash
python Finance/chunk_documents.py --config Finance/config.yaml
```

4. **Index to Qdrant**:
```bash
python Finance/upsert.py --checkpoint-file Finance/upsert_checkpoint.json
```

5. **Run evaluation**:
```bash
python Finance/rubric.py --result-file Finance/result.json
```

### Testing

```bash
python Finance/datatest.py  # Component-level tests
python datatest.py          # Integration tests
```

## Document Collection

The system processes multiple document types:

- Budget Policy Statements (annual)
- Budget Review and Outlook Papers (quarterly reports)
- Economic Surveys (annual analysis)
- Tax Expenditure Reports
- Corporate Plans (state corporations)
- Statistical Annexes (tables and data)
- Sector-specific policies

**Supported periods**: FY 2015-16 through FY 2026-27

## Configuration Details

### Chunking Strategies
- **narrative**: Paragraph-based chunking (350 tokens, 50-token overlap)
- **tables_only**: Row-based chunking for tabular data (200 tokens, no overlap)

### Document Priorities
- **high**: Critical fiscal policy documents (RAG weight: 1.5x)
- **medium**: Supporting documents (RAG weight: 1.0x)
- **low**: Reference materials (RAG weight: 0.5x)

### Review Levels
- **none**: Fully resolved, ready for production
- **soft**: Spot-check recommended (non-critical fields)
- **hard**: Manual review required before indexing

## Vector Database

Collection: `kenya_executive_roundtable`

- Embedding model: SentenceTransformers (configurable)
- Vector dimension: 384 (SentenceTransformer default) or 1024 (VoyageAI)
- Similarity metric: Cosine distance
- Deduplication threshold: 0.85

## LLM Inference

### Current Configuration
- Model: Groq Llama 3.1 70B (free tier)
- Provider: Groq API
- Use case: Development and prototyping

### Post-Fine-Tuning
- Model: Together AI (DeepSeek R1 Distill Qwen 14B)
- Adaptation: LoRA adapter for domain-specific knowledge
- Provider: Together AI API

## Data Caching

Processed document metadata is cached in JSON format for rapid retrieval:

```
Finance/data/cache/
├── 2025_budget_policy_statement.json
├── 2025_economic_survey.json
├── 2024_budget_review_and_outlook_paper.json
└── ... (35+ cached documents)
```

Cache structure enables:
- Fast document reload without re-processing
- Metadata access without full PDF parsing
- Checkpoint-based incremental indexing

## Pipeline Output

Generated artifacts:

- **result.json**: Pipeline execution metrics and performance data
- **upsert_checkpoint.json**: Timestamp and document tracking for incremental updates
- **chunks/**: Segmented document chunks for debugging and analysis

## Quality Assurance

The rubric system evaluates:
- Chunk coherence and relevance
- Deduplication effectiveness
- Embedding quality
- Retrieval relevance
- Answer accuracy on test queries

Run evaluation:
```bash
python Finance/rubric.py --config Finance/config.yaml
```

## Extending to Other Sectors

Each sector (Agriculture, AntiCorruption, Education, ICT, Infrastructure, President) replicates the Finance pipeline structure:

1. Create sector-specific configuration
2. Place source documents in `{Sector}/data/raw/`
3. Instantiate agent with sector identifier
4. Run processing pipeline with sector-specific parameters

## Troubleshooting

### Vector Database Connection Issues
- Verify Qdrant is running: `curl http://localhost:6333/health`
- Check network connectivity to Qdrant server
- Ensure collection exists or allow auto-creation

### Document Processing Failures
- Verify PDF files are readable and not corrupted
- Check character encoding for text extraction
- Review HARD-flagged entries in config.yaml

### Agent Inference Errors
- Verify API keys are set and valid
- Check LLM token usage and rate limits
- Review context length for retrieval results

## Performance Considerations

- **Chunking**: Optimized for semantic boundaries (100-500 tokens)
- **Embedding**: Batch processing for multiple documents
- **Search**: Indexed vector search via Qdrant (sub-second latency)
- **Caching**: JSON metadata cache reduces document re-processing

## Dependencies

Core packages:
- `groq`: LLM inference
- `voyageai`: Embedding generation
- `qdrant-client`: Vector database operations
- `fastembed`: Lightweight embeddings (alternative)
- `sentence-transformers`: Embedding models

## Future Enhancements

- Multi-agent consensus mechanisms for cross-sector queries
- Fine-tuned domain-specific models
- Real-time document ingestion pipeline
- Web interface for executive queries
- Audit trail and explainability features
- Multi-language support

## License

Internal project for Kenya AI Executive Roundtable. Confidential.

## Contact

For questions about pipeline architecture or configuration, refer to Finance module documentation and generated config comments.