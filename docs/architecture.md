# Chat Platform Architecture

## Objective

Build a scalable enterprise document intelligence platform.

The platform should support:

- Multiple document formats
- RAG
- Agent architecture
- Image retrieval
- Table retrieval
- Citations
- Conversation memory

---

## High Level Flow

Upload

↓

Parser

↓

RawDocument

↓

Normalizer

↓

Chunk Generator

↓

Metadata Enrichment

↓

Embedding

↓

Qdrant

↓

Retriever

↓

Agent

↓

LLM

↓

Response

---

## Components

### Parser

Responsible only for parsing documents.

Never performs AI tasks.

Supported parsers:

- PDFParser
- DOCXParser
- PPTXParser
- ExcelParser

---

### Chunk Generator

Responsible for creating semantic chunks.

Must never hardcode document-specific logic.

---

### Metadata

Every chunk contains metadata.

Metadata enables:

- citations
- image retrieval
- table retrieval
- filtering

---

### Retriever

Uses

- vector similarity

and

- metadata filtering

---

### Agent

Responsible for deciding which tool to use.

Initially:

- RAG Tool

Future:

- fetch_images
- fetch_tables
- fetch_database
- search_projects

---

### Chat

Chat never accesses Qdrant directly.

Chat only communicates with Agent.

---

### Memory

Short-term memory

Conversation history.

Long-term memory

User preferences.

---

### API

POST /upload

POST /chat

GET /documents

GET /projects

GET /health

---

## Future Enhancements

OCR

Multilingual support

Streaming responses

Hybrid Search

Re-ranking

Multiple agents

Caching

Authentication