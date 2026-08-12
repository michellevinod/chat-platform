# Chat Platform - Copilot Instructions

## Project Overview

This project is an enterprise Document Intelligence Platform with a conversational interface.

The system must work with ANY domain and ANY document type.

Examples:
- Reports
- Bills
- Invoices
- Contracts
- College Notes
- Books
- Research Papers
- Manuals
- SOPs
- Resumes
- Presentations
- Excel sheets

Never assume a specific domain.

---

## Supported Document Types

Current:
- PDF

Upcoming:
- DOCX
- PPTX
- XLSX
- TXT
- Markdown

Every document type must pass through the SAME pipeline.

---

## Architecture

Upload

↓

Parser

↓

Generic Document Model

↓

Normalizer

↓

Chunk Generator

↓

Metadata Enrichment

↓

Embedding Generation

↓

Qdrant

↓

Search Layer

↓

Chat Agent

---

## Coding Rules

- Follow SOLID principles.
- Follow clean architecture.
- Prefer composition over inheritance.
- Never hardcode values.
- Never hardcode headings.
- Never hardcode document names.
- Never hardcode project names.
- Never hardcode IDs.
- Never assume document structure.
- Never assume language.
- Never assume file type.
- Never assume domain.

Everything should be configurable.

---

## Generic Models

All parsers MUST return RawDocument.

No parser should return library-specific objects.

Examples:
- Never expose PyMuPDF classes.
- Never expose python-docx classes.
- Never expose openpyxl classes.

Everything should be converted into project models.

---

## Repositories

Repositories only communicate with databases.

Repositories must never contain business logic.

---

## Services

Business logic belongs only inside services.

---

## Qdrant

Qdrant is the ONLY vector database.

Every point should contain:

- embedding
- chunk text
- metadata

Metadata should support filtering.

---

## Metadata

Metadata must be generic.

Examples:

project_id

project_name

document_id

document_name

page_number

chunk_number

document_type

section

contains_table

contains_image

language

source

No domain-specific metadata.

---

## Search

Search should combine:

- Semantic similarity
- Metadata filtering

---

## Chat

The assistant only answers questions related to uploaded documents.

If user asks unrelated questions:

Politely refuse.

Example:

"What is the capital of France?"

↓

"I'm designed to answer questions only from uploaded documents."

Greetings should NOT invoke RAG.

---

## Security

Never expose:

- Internal IDs
- Vector IDs
- File system paths
- Database IDs

Only expose citations.

---

## Code Style

- Use type hints everywhere.
- Use Pydantic models.
- Write reusable functions.
- Write docstrings.
- Keep functions small.
- Avoid duplicate code.
- No magic numbers.
- No global state.
- Use dependency injection where appropriate.