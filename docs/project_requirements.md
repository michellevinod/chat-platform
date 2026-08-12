# Functional Requirements

The system should support:

## Ingestion

- PDF
- DOCX
- PPTX
- XLSX

Generic ingestion pipeline.

---

## Chat

Answer only based on uploaded documents.

Never answer general knowledge.

Greetings should be answered instantly.

---

## RAG

Use only Qdrant.

Embeddings + metadata.

---

## Images

Support retrieving images associated with relevant answers.

---

## Tables

Support retrieving table information.

---

## Citations

Every response should include

Document Name

Page Number

Future:

Section

---

## Memory

Short-term

Long-term

---

## Agent

Initially:

One agent.

Future:

Multiple specialized agents.

---

## Security

Never expose:

Internal IDs

Vector IDs

Database IDs

Server paths

---

## Performance

Responses should be fast.

Avoid unnecessary tool calls.

Greetings should not invoke retrieval.

---

## Scalability

Support:

1000+ page documents

Large projects

Multiple uploaded documents

Multiple projects

Multiple users