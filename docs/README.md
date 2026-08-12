# Chat Platform

An enterprise-grade AI-powered Document Intelligence Platform that enables users to upload documents, semantically search their contents, and interact with them through a conversational interface.

The platform is designed to support multiple document formats through a unified ingestion pipeline while providing accurate Retrieval-Augmented Generation (RAG), metadata-aware retrieval, citations, and extensible agent-based workflows.

---

# Features

## Document Ingestion

- PDF support
- Extensible parser architecture
- Generic ingestion pipeline
- Layout-aware document processing
- Metadata extraction
- Semantic chunk generation
- Embedding generation

Upcoming Support

- DOCX
- PPTX
- XLSX
- TXT
- Markdown
- OCR for scanned documents

---

## Semantic Search

- Retrieval-Augmented Generation (RAG)
- Vector similarity search using Qdrant
- Metadata-aware retrieval
- Multi-document search
- Project-level filtering
- Document-level filtering

---

## Conversational AI

- Chat-based document interaction
- Context-aware retrieval
- Follow-up question support (planned)
- Short-term conversation memory (planned)
- Long-term memory (planned)

---

## Rich Responses

Planned support for returning:

- Text
- Tables
- Images
- Citations
- Document references
- Page references

---

## Security

- Internal IDs are never exposed
- Vector database IDs remain hidden
- Read-only database access
- Metadata-safe responses

---

# Architecture

```
                    Upload
                       │
                       ▼
                File Detection
                       │
        ┌──────────────┼──────────────┐
        │              │              │
      PDF           DOCX           PPTX
        │              │              │
        └──────────────┼──────────────┘
                       │
                 Parser Layer
                       │
                       ▼
               Generic Document Model
                       │
                       ▼
                  Normalization
                       │
                       ▼
               Semantic Chunking
                       │
                       ▼
              Metadata Enrichment
                       │
                       ▼
              Embedding Generation
                       │
                       ▼
                    Qdrant
                       │
                       ▼
                Retrieval Layer
                       │
                       ▼
                   Chat Agent
                       │
                       ▼
                  Final Response
```

---

# Technology Stack

## Backend

- Python
- FastAPI
- Pydantic

## AI

- Sentence Transformers
- Retrieval-Augmented Generation (RAG)

## Vector Database

- Qdrant

## Document Processing

- PyMuPDF
- python-docx *(planned)*
- python-pptx *(planned)*
- openpyxl *(planned)*

---

# Project Structure

```
app/
│
├── agents/
├── api/
├── chunking/
├── embeddings/
├── ingestion/
│   ├── extractors/
│   ├── normalizers/
│   ├── metadata/
│   └── sectioning/
│
├── models/
├── prompts/
├── rag/
├── repositories/
├── schemas/
├── services/
├── storage/
└── utils/

tests/
```

---

# Workflow

1. Upload document
2. Detect document type
3. Parse document
4. Normalize extracted content
5. Generate semantic chunks
6. Enrich metadata
7. Generate embeddings
8. Store vectors in Qdrant
9. Retrieve relevant chunks
10. Generate response with citations

---

# Current Progress

## Completed

- Generic ingestion architecture
- PDF parsing
- Document normalization
- Metadata generation
- Chunk generation
- Embedding generation
- Qdrant integration
- Semantic search
- Modular project structure

## In Progress

- Universal parser support
- Improved semantic chunking
- Agent framework
- Citation formatting
- Image retrieval
- Table retrieval
- Conversation memory
- Chat orchestration

## Planned

- DOCX support
- PPTX support
- XLSX support
- OCR support
- Hybrid search
- Re-ranking
- Multi-agent architecture
- Streaming responses
- Authentication & authorization

---

# API (Planned)

## Upload Document

```
POST /upload
```

Uploads a supported document and starts the ingestion pipeline.

---

## Chat

```
POST /chat
```

Answers questions based only on uploaded documents.

---

## Documents

```
GET /documents
```

Lists uploaded documents.

---

## Projects

```
GET /projects
```

Lists available projects.

---

## Health

```
GET /health
```

Returns service health status.

---

# Design Principles

- Clean Architecture
- SOLID Principles
- Modular Design
- Reusable Components
- Extensible Parser Framework
- Metadata-Driven Retrieval
- Domain-Agnostic Processing
- Dependency Injection
- Separation of Concerns

---

# Goals

The platform is designed to:

- Support multiple document formats
- Handle large documents (1000+ pages)
- Provide fast semantic retrieval
- Return context-aware responses
- Support citations
- Retrieve text, tables, and images
- Scale to enterprise document collections
- Enable future multi-agent workflows

---

# License

This project is intended for internal development and research purposes.