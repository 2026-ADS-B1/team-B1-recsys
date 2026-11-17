# Recommendation API (US6.1 + US6.2, Sprint 1)

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

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Says if the service is running |
| GET | `/genres` | Lists the genres you can ask for |
| POST | `/recommend` | Returns the recommended movies |

### POST /recommend

Every field is optional:

```json
{"user_id": 1, "genre": "Comedy", "n": 5}
```

- `user_id` given: the user must exist; movies they already rated are left out.
- `user_id` missing: a new user, who gets the best-rated movies.
- `genre` given: only movies of that genre (upper or lower case both work).
- `n`: how many movies (1 to 50, default 5).
- `mood`: reserved for a later sprint; it answers 501 for now.

Answer:

```json
{
  "user_id": 1,
  "genre": "Comedy",
  "source": "popularity-baseline",
  "recommendations": [
    {"rank": 1, "movie_id": 1, "title": "Toy Story (1995)", "genres": ["Animation", "Children's", "Comedy"], "score": 4.12}
  ]
}
```

Errors: `404` user not in the system, `422` genre that doesn't exist (the message lists the valid ones)
or invalid request, `501` mood.

## Tests

```bash
pytest tests/api
```

They use small fake data, so they don't need the real files.
