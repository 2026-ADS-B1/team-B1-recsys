# Recommendation API (US6.1, Sprint 1)

Small FastAPI service: you send a user, a genre or both, and it answers with the top movies
ordered from most to least recommended. It uses the team's popularity baseline
(`src/models/baseline.py`, `compute_movie_stats`).

## Start it

From the root of the repository (Python 3.10 or newer):

```bash
pip install -r api/requirements.txt
dvc pull                       # or copy MovieLens 1M into data/raw/ml-1m/
uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000/docs to see and try every endpoint.

The service reads `data/raw/ml-1m/ratings.dat` and `data/raw/ml-1m/movies.dat` by default.
To use other files (for example the Parquet files of the pipeline), set:

| Variable | Default | Meaning |
|---|---|---|
| `RATINGS_PATH` | `data/raw/ml-1m/ratings.dat` | Ratings (`.dat`, `.csv` or `.parquet`) |
| `MOVIES_PATH` | `data/raw/ml-1m/movies.dat` | Movies with title and genres |
| `MIN_RATINGS` | `50` | Minimum ratings a movie needs to be recommended |

## POST /recommend

```json
{"user_id": 1, "n": 5}
```

Returns the `n` best-rated movies the user hasn't rated yet, with title and genres.
An unknown user returns `404` with a clear message. Genre comes in the next task.

## Tests

```bash
pytest tests/api
```
