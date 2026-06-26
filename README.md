# Research Agent

> An AI-powered academic research assistant built with Flask and IBM Granite — automating literature discovery, citation analysis, trend prediction, knowledge graph extraction, and insight generation.

---

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Folder Structure](#folder-structure)
- [API Endpoints](#api-endpoints)
- [Screenshots](#screenshots)
- [Future Scope](#future-scope)
- [License](#license)

---

## Description

**Research Agent** is a full-stack, AI-powered research assistant that automates the most time-consuming parts of academic research. Upload a PDF, enter a research topic, and the system:

1. Extracts and indexes the document into a FAISS vector store (RAG pipeline)
2. Retrieves relevant context for every query
3. Routes the query through a 5-agent pipeline powered by IBM Granite
4. Returns a comprehensive, structured research report

---

## Features

| Feature | Description |
|---|---|
| **PDF Upload & Indexing** | Upload PDFs → PyMuPDF extraction → chunking → FAISS embeddings |
| **AI Chat** | RAG-grounded Q&A against your uploaded documents |
| **Literature Review** | Auto-generates 6-section academic literature reviews |
| **Citation Analysis** | Identifies key references, missing citations, and influential papers |
| **Trend Prediction** | Forecasts emerging topics, future directions, and research opportunities |
| **Knowledge Graph** | Extracts authors, institutions, concepts, methods, keywords, and datasets as structured JSON |
| **Insight Generation** | Surfaces research gaps, novel ideas, future work, and recommendations |
| **Full Pipeline Orchestration** | Runs all 5 agents sequentially and returns a unified report |
| **Report Download** | Export the complete pipeline output as a `.txt` report |
| **Interactive Dashboard** | Bootstrap 5 SPA with Plotly.js charts and sidebar navigation |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | Flask 3.x (Python) |
| **LLM / AI** | IBM watsonx.ai — IBM Granite (`granite-4-h-small`) |
| **RAG Pipeline** | PyMuPDF · FAISS · IBM watsonx Embeddings |
| **Vector Store** | FAISS (persisted to disk) |
| **Frontend** | Bootstrap 5 · Bootstrap Icons · Plotly.js |
| **Templating** | Jinja2 |
| **Configuration** | python-dotenv |
| **Dependency Mgmt** | pip / requirements.txt |

---

## Project Architecture

```
User Browser
     │
     ▼
Flask Application (app.py)
     │
     ├── main blueprint      → Landing page  (/)
     ├── dashboard blueprint → Dashboard SPA (/dashboard)
     └── api blueprint       → REST JSON API (/api/*)
               │
               ├── POST /api/upload      → RAG Pipeline
               │       │
               │       └── PDFLoader → TextChunker → Embedder → VectorStore (FAISS)
               │
               ├── POST /api/chat        → WatsonxClient (direct Q&A)
               │
               └── POST /api/orchestrate → AgentOrchestrator
                           │
                           ├── [1] LiteratureReviewAgent   (6 sections)
                           ├── [2] CitationAnalysisAgent   (4 sections)
                           ├── [3] TrendPredictionAgent    (4 sections)
                           ├── [4] KnowledgeGraphAgent     (JSON entities)
                           └── [5] InsightGenerationAgent  (4 sections)
                                       │
                                       ▼
                               IBM Granite (via WatsonxClient)
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip
- Git
- An IBM Cloud account with a watsonx.ai project

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/Dasabinash150/Research-Agent.git
cd Research-Agent
```

**2. Create and activate a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` and fill in your IBM credentials (see [Environment Variables](#environment-variables)).

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `IBM_API_KEY` | ✅ Yes | IBM Cloud API key |
| `IBM_PROJECT_ID` | ✅ Yes | watsonx.ai project GUID |
| `IBM_URL` | ✅ Yes | watsonx.ai service URL (region-specific) |
| `MODEL_ID` | ✅ Yes | Foundation model identifier (e.g. `granite-4-h-small`) |
| `EMBEDDING_MODEL` | ⚙️ Optional | Embedding model ID (defaults to `ibm/slate-125m-english-rtrvr`) |
| `SECRET_KEY` | ✅ Yes | Flask session secret key — use a long random string |
| `FLASK_DEBUG` | ⚙️ Optional | Set to `true` for development hot-reload |

**`.env` template** (copy from `.env.example`):

```dotenv
IBM_API_KEY=YOUR_API_KEY
IBM_PROJECT_ID=YOUR_PROJECT_ID
IBM_URL=https://us-south.ml.cloud.ibm.com
MODEL_ID=granite-4-h-small
EMBEDDING_MODEL=YOUR_EMBEDDING_MODEL
SECRET_KEY=YOUR_SECRET_KEY
```

> ⚠️ Never commit `.env` to version control. It is listed in `.gitignore`.

---

## Running the Application

```bash
python app.py
```

The server starts at **http://127.0.0.1:5000**

| URL | Description |
|---|---|
| `http://localhost:5000/` | Landing page |
| `http://localhost:5000/dashboard` | Research dashboard |
| `http://localhost:5000/api/dashboard` | API health & stats (JSON) |

---

## Folder Structure

```
Research-Agent/
│
├── agents/                          # AI agent modules
│   ├── __init__.py
│   ├── base_agent.py                # Abstract base class for all agents
│   ├── watsonx_client.py            # IBM Granite SDK wrapper
│   ├── literature_review_agent.py   # 6-section literature review
│   ├── citation_analysis_agent.py   # Citation landscape analysis
│   ├── trend_prediction_agent.py    # Trend & opportunity forecasting
│   ├── knowledge_graph_agent.py     # Entity extraction → JSON
│   ├── insight_generation_agent.py  # Research gaps & novel ideas
│   └── orchestrator.py              # 5-agent sequential pipeline
│
├── rag/                             # Retrieval-Augmented Generation
│   ├── __init__.py
│   ├── pdf_loader.py                # PyMuPDF text extraction
│   ├── chunker.py                   # Sliding-window text chunker
│   ├── embedder.py                  # IBM / local sentence-transformers
│   └── vector_store.py              # FAISS index + persistence
│
├── api/                             # REST API blueprint
│   ├── __init__.py
│   └── routes.py                    # All 8 JSON endpoints
│
├── main/                            # Public page blueprint
│   ├── __init__.py
│   └── routes.py                    # Landing page route
│
├── templates/                       # Jinja2 HTML templates
│   ├── base.html                    # Bootstrap 5 base layout
│   ├── index.html                   # Landing page
│   └── dashboard.html               # Research dashboard SPA
│
├── static/
│   ├── css/
│   │   ├── style.css                # Global styles
│   │   └── dashboard.css            # Dashboard-specific styles
│   ├── js/
│   │   └── dashboard.js             # Plotly charts + all API calls
│   └── images/
│
├── uploads/                         # PDF uploads (git-ignored)
├── vector_db/                       # FAISS index files (git-ignored)
├── reports/                         # Generated reports (git-ignored)
├── logs/                            # Application logs (git-ignored)
│
├── app.py                           # Application factory & entry point
├── config.py                        # Dev / Prod / Test config classes
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
├── .gitignore                       # Git exclusion rules
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description | Body |
|---|---|---|---|
| `GET` | `/api/dashboard` | Health check & stats | — |
| `POST` | `/api/upload` | Ingest a PDF into FAISS | `multipart/form-data` — field `file` |
| `POST` | `/api/chat` | RAG-grounded Q&A | `{ "query": "..." }` |
| `POST` | `/api/summary` | Literature review | `{ "query": "..." }` |
| `POST` | `/api/citation` | Citation analysis | `{ "query": "..." }` |
| `POST` | `/api/trend` | Trend prediction | `{ "query": "..." }` |
| `POST` | `/api/knowledge` | Knowledge graph extraction | `{ "query": "..." }` |
| `POST` | `/api/insight` | Insight generation | `{ "query": "..." }` |
| `POST` | `/api/orchestrate` | Full 5-agent pipeline | `{ "query": "..." }` |

All endpoints return:
```json
{ "ok": true,  "data": { ... } }   // success
{ "ok": false, "error": "..." }    // failure
```

---

## Screenshots

> _Add screenshots of the dashboard, chat interface, and output panels here._

| Page | Screenshot |
|---|---|
| Landing Page | _(add screenshot)_ |
| Dashboard — Overview | _(add screenshot)_ |
| Literature Review Output | _(add screenshot)_ |
| Knowledge Graph Entities | _(add screenshot)_ |
| Trend Analysis Chart | _(add screenshot)_ |

---

## Future Scope

| Feature | Description |
|---|---|
| **Multi-document RAG** | Index and query across multiple PDFs simultaneously |
| **Citation Graph Visualisation** | Interactive D3.js or Cytoscape.js knowledge graph rendering |
| **Streaming Responses** | Server-Sent Events for real-time token streaming to the dashboard |
| **User Authentication** | Flask-Login / JWT for multi-user support |
| **Export Formats** | PDF and DOCX report export (in addition to `.txt`) |
| **Agent Memory** | Persistent conversation history per research session |
| **Web Search Integration** | Supplement RAG with live web retrieval (Tavily / SerpAPI) |
| **Fine-tuned Models** | Domain-specific Granite fine-tunes for medical/legal research |
| **Docker Deployment** | Containerised deployment with `docker-compose` |
| **CI/CD Pipeline** | GitHub Actions workflow for automated testing and deployment |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Abinash Das

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">Built with IBM watsonx.ai · Flask · FAISS · Bootstrap 5</p>
