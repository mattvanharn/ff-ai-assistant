# Fantasy Football AI Draft Assistant

A hybrid AI draft assistant that combines a RAG (Retrieval-Augmented Generation) knowledge base with an analytics engine to answer natural language fantasy football questions — grounded in real stats, ADP data, and expert analysis.

## What It Does

Ask questions like:
- "Round 4, pick 8 — I have 2 RBs and 1 WR, who should I take?"
- "Is Ja'Marr Chase worth his ADP this year?"
- "What are the injury concerns for Saquon Barkley?"
- "Who are the best value picks at TE after round 5?"

The system retrieves relevant expert analysis from a vector database and cross-references it with real stats and ADP value calculations — then synthesizes both into an answer with reasoning and source citations.

## Architecture

```
Question + Roster + Drafted Players
          ↓
┌─────────┴─────────┐
│                   │
RAG Layer       Analytics Engine
│                   │
Expert opinions,   Stats, ADP,
injury context,    value scores,
news, narratives   positional needs
│                   │
└─────────┬─────────┘
          ↓
     LLM synthesis
          ↓
  Answer + Reasoning + Sources
```

## Tech Stack

| Component | Tool |
|-----------|------|
| RAG framework | LangChain |
| Vector database | ChromaDB |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM | Groq API (`llama-3.1-70b`) |
| Player stats | nflreadpy |
| ADP data | Sleeper API |
| Testing | pytest |
| Python version | 3.11 via pyenv |

## Setup

```bash
# Clone the repo
git clone https://github.com/mattvanharn/fantasy-draft-rag.git
cd fantasy-draft-rag

# Set Python version (requires pyenv)
pyenv local 3.11.14

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)
```

## Project Status

**Phase 1: Hybrid RAG + Analytics Pipeline** — In Progress

| Step | Status | Description |
|------|--------|-------------|
| 1. Project setup | ✅ Done | Repo, venv, dependencies |
| 2. Groq LLM setup | ⬜ | Configure LLM via API |
| 3. Stats data | ⬜ | Collect 2024 player stats |
| 4. ADP data | ⬜ | Collect ADP and rankings |
| 5. Document processing | ⬜ | Convert stats to text for RAG |
| 6. Vector store | ⬜ | Embeddings + ChromaDB |
| 7. Analytics engine | ⬜ | Value scores, roster logic |
| 8. End-to-end pipeline | ⬜ | Connect all components |
| 9. Evaluation & polish | ⬜ | Testing, tuning, docs |

## License

MIT
