# Recommendation API (US6.1, Sprint 1)

FastAPI service that answers movie recommendations. For now `POST /recommend`
returns a fixed example answer (task #25); real results come in the next task.

## Start it

From the root of the repository (Python 3.10 or newer):

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000/docs to see the request and response examples.

## Tests

```bash
pytest tests/api
```
