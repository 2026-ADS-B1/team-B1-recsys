# Recommendation API (US6.1 + US6.2, Sprint 1)

Small FastAPI service: you send a user, a genre or a mood, and it answers with the top movies
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

## Settings (`params.yaml`)

The service reads its settings from the `serving` section of `params.yaml`, at the
root of the repository. Change them there, not in the code:

```yaml
serving:
  min_ratings: 50   # a movie needs at least 50 ratings to be recommended
```

Only movies with at least `min_ratings` ratings are recommended, to all users, so
that a movie with a single 5-star rating doesn't top the list. New users (no
`user_id`) get the best-rated of those movies. For a quick test you can override it
with an environment variable: `MIN_RATINGS=20 uvicorn api.main:app`.
Set `PARAMS_PATH` to use another settings file.

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
at least `min_ratings` ratings are scored.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Says if the service is running and which model is loaded |
| GET | `/genres` | Lists the genres you can ask for |
| GET | `/moods` | Lists the moods you can ask for and their genres |
| POST | `/recommend` | Returns the recommended movies |

### POST /recommend

Every field is optional:

```json
{"user_id": 1, "genre": "Comedy", "n": 5}
{"mood": "relaxed"}
```

- `user_id` given: the user must exist; movies they already rated are left out.
- `user_id` missing: a new user, who gets the best-rated movies.
- `genre` given: only movies of that genre (upper or lower case both work).
- `mood` given: only movies that fit that mood (see the list below). Send a genre **or** a mood, not both.
- `n`: how many movies (1 to 50, default 5).

Answer:

```json
{
  "user_id": 1,
  "genre": "Comedy",
  "mood": null,
  "source": "popularity-baseline",
  "recommendations": [
    {"rank": 1, "movie_id": 1, "title": "Toy Story (1995)", "genres": ["Animation", "Children's", "Comedy"], "score": 4.12}
  ]
}
```

Errors: `404` user not in the system; `422` genre or mood that doesn't exist (the message
lists the valid ones), a genre and a mood sent together, or an invalid request.

### Moods

Moods are defined in `params.yaml` (`serving.moods`). A movie fits a mood if it has
at least one of the mood's genres. To add or change a mood, edit `params.yaml` and
restart the service (genre names must be written as in `GET /genres`).

| Mood | Genres |
|---|---|
| `relaxed` | Comedy, Animation, Children's, Musical |
| `adventurous` | Action, Adventure, Sci-Fi, Fantasy, Western |
| `romantic` | Romance |
| `thrilling` | Thriller, Crime, Mystery, Film-Noir |
| `scary` | Horror |
| `thoughtful` | Drama, Documentary, War |

## Tests

```bash
pytest tests/api
```

They use small fake data, so they don't need the real files.
