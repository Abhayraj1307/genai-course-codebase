# Generative AI Course Codebase

Coursework and experiments with Flask chat interfaces, agent patterns, and retrieval-augmented generation (RAG). The folders capture learning exercises and examples.

- [Flask chatbot](flask-chatbot/)
- [Flask agent](flask-agent/)
- [RAG notes](rag.md)

## Run the chatbot locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python flask-chatbot/app.py
```

Configure the provider settings locally before making model requests. The RAG examples use Qdrant; see the [retrieval notes](rag.md) for indexing details. Keep provider keys out of version control.
