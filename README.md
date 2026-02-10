# RAG-based Examination Regulations Chatbot

A Retrieval Augmented Generation (RAG) system for answering questions about university examination regulations at the University of Bamberg.

## Overview

This system combines document retrieval with large language model generation to provide accurate, cited answers to student questions about examination regulations, policies, and procedures.

### Key Features

- **Multi-source ingestion**: PDFs, FAQ data, and department routing information
- **Semantic search**: Dense vector retrieval with MMR for diverse results
- **Grounded answers**: Strict citation requirements with source references
- **Department routing**: Automatically directs students to appropriate university offices
- **Multiple interfaces**: CLI and Streamlit web UI
- **Evaluation framework**: RAGAS metrics for performance assessment

## Architecture

```
v1/
├── core/           # Configuration and logging
├── ingestion/      # Document processing and preprocessing
├── retrieval/      # Vector store and retrieval engine
├── generation/     # LLM interface and prompts
├── evaluation/     # RAGAS evaluation
└── ui/             # CLI and Streamlit interfaces
```

## Prerequisites

- **Python 3.12+** (required)
- Access to University Ollama server (http://ursa.ds.uni-bamberg.de:11434)
- PDF regulation documents
- FAQ data (QA.json)
- Department information (departments.json)

## Setup

### 1. Verify Python Version

```bash
python3.12 --version
# Should output: Python 3.12.x
```

### 2. Create Virtual Environment

```bash
python3.12 -m venv venv
```

### 3. Activate Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure System

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` with your settings (Ollama URL, model preferences, etc.)

### 6. Prepare Data

Ensure the following files/directories exist:
- `Studienordnungen/` - Directory containing PDF regulation documents
- `QA.json` - FAQ data from university website
- `departments.json` - Department routing information

## Usage

### Ingestion

Process documents and populate the vector database:

```bash
python v1/ingest.py
```

This will:
- Extract text from PDFs
- Process FAQ and department data
- Generate embeddings
- Store in Chroma vector database

### CLI Chat

Interactive command-line interface:

```bash
python v1/chat.py
```

Type your questions and receive streaming responses. Type `exit` or `quit` to end.

### Streamlit UI

Web-based chat interface:

```bash
streamlit run v1/ui.py
```

With debug mode (shows sources, R1 thinking tokens, and debug info):

```bash
streamlit run v1/ui.py -- --debug
```

Access at http://localhost:8501

### Evaluation

Run RAGAS evaluation on test dataset:

```bash
python v1/evaluate.py
```

Results will be saved to CSV and Markdown reports.

## Configuration

Key configuration options in `config.yaml`:

- **Generation Model**: `deepseek-r1:70b` (default), `deepseek-r1:14b`, `llama3.1:70b`, `mistral:7b`
- **Embedding Model**: `bge-m3:latest` (default), `mxbai-embed-large:latest`, `nomic-embed-text`
- **Retrieval Parameters**: `fetch_k`, `final_k`, `search_type`
- **Chunking**: `chunk_size`, `chunk_overlap`

## Example Queries

- "Wie viele ECTS brauche ich für den Bachelor?"
- "Kann ich mich für eine Prüfung abmelden?"
- "Ich möchte ein Auslandssemester machen, wie funktioniert die Anerkennung?"
- "Wie beantrage ich ein Urlaubssemester?"

## Troubleshooting

### Ollama Connection Issues

If you get connection errors:
1. Verify the Ollama server URL in `config.yaml`
2. Test connectivity: `curl http://ursa.ds.uni-bamberg.de:11434/api/tags`
3. Check if you're on the university network or VPN

### Import Errors

Make sure your virtual environment is activated:
```bash
source venv/bin/activate  # macOS/Linux
```

### Chroma Database Issues

If you encounter database errors, try deleting and recreating:
```bash
rm -rf chroma_db/
python v1/ingest.py
```

## Project Structure

```
.
├── v1/                          # Main application code
│   ├── core/                    # Configuration and logging
│   ├── ingestion/               # Document processing
│   ├── retrieval/               # Vector store and search
│   ├── generation/              # LLM and prompts
│   ├── evaluation/              # RAGAS metrics
│   └── ui/                      # User interfaces
├── Studienordnungen/            # PDF documents
├── chroma_db/                   # Vector database (generated)
├── QA.json                      # FAQ data
├── departments.json             # Department routing info
├── config.yaml                  # Configuration (create from example)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Development

This project follows a modular architecture:
- Each subdirectory in `v1/` handles a specific concern
- Configuration is centralized in `config.yaml`
- Logging is verbose for debugging
- All components are designed to be testable

