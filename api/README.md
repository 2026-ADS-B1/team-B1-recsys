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

## Which model is used

The model is chosen with environment variables when the service starts. To use a
different model, change the variables (or promote another model in MLflow) and
**restart the service**: no code changes are needed.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_SOURCE` | `baseline` | `baseline` = popularity baseline computed at startup. `mlflow` = model from the MLflow registry |
| `MODEL_NAME` | `recommender` | Registered model name (only with `MODEL_SOURCE=mlflow`) |
| `MODEL_ALIAS` | `champion` | Alias of the version to load, e.g. `champion` or `challenger` |
| `MLFLOW_TRACKING_URI` | – | Where the MLflow server is, e.g. `http://localhost:5000` |

`GET /health` shows the model that is loaded, for example:

```json
{"status": "ok", "model": {"source": "mlflow", "name": "recommender", "alias": "champion", "version": "3"}}
```

Example with MLflow (needs `pip install mlflow` and a registered model):

```bash
MODEL_SOURCE=mlflow MLFLOW_TRACKING_URI=http://localhost:5000 uvicorn api.main:app
```

Model contract for MLflow (to agree with US4.1): a pyfunc model that receives a
DataFrame with a `movie_id` column and returns one score per row. Only movies with
at least `MIN_RATINGS` ratings are scored.

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
