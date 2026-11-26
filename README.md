# Movie Recommendation System (`team-B1-recsys`)

ML-based movie recommendation system built on the MovieLens 1M dataset, developed following Agile Data Science practices.

---

## Sprint 1 Overview & Architecture

In Sprint 1, the end-to-end recommendation pipeline and user interface were established across multiple epics:

```
┌────────────────────────────────────────────────────────┐
│              Streamlit Web App (Epic 7 / US 7.2)        │
│          • User ID / Guest Mode                        │
│          • Genre or Mood Selector (Mutually Exclusive) │
│          • Ranked Movie Results & Friendly Feedback    │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP (POST /recommend)
                            ▼
┌────────────────────────────────────────────────────────┐
│            FastAPI Recommendation API (Epic 6)         │
│          • Endpoints: /recommend, /genres, /moods      │
│          • Config: params.yaml (moods, min_ratings)    │
└───────────────────────────┬────────────────────────────┘
                            │ Scores & Filters
                            ▼
┌────────────────────────────────────────────────────────┐
│        Baseline Recommender Engine (Epic 2)            │
│          • Popularity-based ranking with min ratings   │
│          • Excludes already rated movies for users     │
└───────────────────────────┬────────────────────────────┘
                            │ Reads raw data
                            ▼
┌────────────────────────────────────────────────────────┐
│          Data Layer (DVC + DagsHub) (Epic 3)           │
│          • MovieLens 1M tracked via pointer files      │
└────────────────────────────────────────────────────────┘
```

---

## Web Application (Epic 7 / US 7.2)

The web application allows users to request personalized movie recommendations directly through an interactive web interface without manually interacting with the REST API.

### Key Features

1. **User Identification:**
   * **Registered Users:** Enter a MovieLens User ID to receive recommendations tailored to their history (movies already rated are automatically filtered out).
   * **Guest Mode:** Check *"Continue as guest"* to browse recommendations as a new user with no past ratings (served the highest-rated movies satisfying the minimum rating threshold).

2. **Filter Preferences (Mutual Exclusion):**
   * Filter recommendations **By Genre** or **By Mood**, or choose **No filter**.
   * Only one filter can be active at a time to avoid conflicting constraints.
   * **Genres** are extracted dynamically from the movie catalogue.
   * **Moods** are read dynamically from `params.yaml` (e.g. *relaxed*, *adventurous*, *thoughtful*, *scary*, etc.).

3. **Real-Time Integration & Feedback:**
   * Connects directly to `POST /recommend` with a visual waiting spinner (`st.spinner`).
   * Displays movies in the exact ranked order received, showing **Rank**, **Title**, **Genres**, and **Score**.

4. **Friendly User Experience & Error Handling (AC4 & AC5):**
   * **Unregistered User ID (404):** Displays a friendly message explaining that the User ID was not found and offers a single-click button to continue as a guest.
   * **Service Unavailable:** Informs the user when the backend API is not running, with instructions on how to start it.
   * **Service Timeout:** Informs the user if the service takes too long to respond, suggesting a retry.

---

## How to Run the Project

Running the full application requires starting both the **FastAPI recommendation backend** and the **Streamlit web frontend**.

### 1. Environment Setup

From the root of the repository (Python 3.10+):

```bash
# Create and activate virtual environment
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# Install dependencies
pip install -r api/requirements.txt
pip install -r app/requirements.txt
pip install -r src/tracking/requirements.txt   # MLflow, to log trainings (see Experiment Tracking)
```

### 2. Download the Dataset

The project requires the MovieLens 1M dataset in `data/raw/ml-1m/`. If you have DagsHub access:

```bash
dvc pull
```
*(Alternatively, extract MovieLens 1M directly into `data/raw/ml-1m/` with `movies.dat` and `ratings.dat`).*

### 3. Start the Backend API (Terminal 1)

```bash
uvicorn api.main:app --reload
```
* The API will be available at: `http://127.0.0.1:8000`
* Interactive API documentation: `http://127.0.0.1:8000/docs`
* Health check: `http://127.0.0.1:8000/health`

### 4. Start the Web Application (Terminal 2)

In a separate terminal:

```bash
streamlit run app/main.py
```
* The browser will automatically open at: `http://localhost:8501`

---

## Configuration (`params.yaml`)

Application and serving settings are defined in `params.yaml` at the root of the repository so they can be modified without altering source code:

```yaml
serving:
  min_ratings: 50   # Minimum ratings a movie needs to be recommended

  moods:            # Mood definitions mapping to MovieLens genres
    relaxed: [Comedy, Animation, Children's, Musical]
    adventurous: [Action, Adventure, Sci-Fi, Fantasy, Western]
    romantic: [Romance]
    thrilling: [Thriller, Crime, Mystery, Film-Noir]
    scary: [Horror]
    thoughtful: [Drama, Documentary, War]

tracking:                                # Shared MLflow server (see Experiment Tracking)
  uri: "https://dagshub.com/e26-kairi/team-B1-recsys.mlflow"  # or MLFLOW_TRACKING_URI env var
  experiment: "recommender"                                    # or MLFLOW_EXPERIMENT_NAME env var

app:
  service_url: "http://127.0.0.1:8000"  # API service address (can be overridden with SERVICE_URL env var)
```

---

## Testing

Run all unit and integration tests across the API, recommendation models, and web application:

```bash
# Run all tests
pytest

# Run only Web App tests
pytest tests/app

# Run only API tests
pytest tests/api

# Run only Model baseline tests
pytest tests/models

# Run only Experiment tracking tests (no network: they use a temporary local MLflow store)
pytest tests/tracking
```

---

## Data Management & DVC (Epic 3)

The raw data is **not stored in Git**. It lives in shared DagsHub storage and is versioned with [DVC](https://dvc.org). Git only tracks the small pointer file (`data/raw/ml-1m.dvc`).

| | |
|---|---|
| Shared storage | DagsHub repository [`e26-kairi/team-B1-recsys`](https://dagshub.com/e26-kairi/team-B1-recsys) |
| DVC remote | `origin` → `https://dagshub.com/e26-kairi/team-B1-recsys.dvc` |
| Local folder | `data/raw/ml-1m/` (git-ignored) |

### 1. Get Access (Once)

1. Sign in to [dagshub.com](https://dagshub.com) with your GitHub account.
2. Ask the owner of the DagsHub repository (`e26-kairi`) to add you as a collaborator with **Write** permission, and accept the invitation.
3. Generate your personal token: DagsHub → your avatar → **Your settings** → **Tokens**.

### 2. Save Credentials (Once per clone)

```bash
dvc remote modify origin --local user <your-dagshub-username>
dvc remote modify origin --local password <your-dagshub-token>
```
*The `--local` flag stores credentials in `.dvc/config.local`, which is git-ignored.*

### 3. Check Dataset Integrity

```bash
dvc status        # Must print: Data and pipelines are up to date.
```

Expected MD5 checksums:

| File | MD5 |
|---|---|
| `movies.dat` | `ef9fd9ff8d1b33faf73e2122c371e910` |
| `ratings.dat` | `a89aa3591bc97d6d4e0c89459ff39362` |
| `users.dat` | `681c0711d0d4548cd3d4718a49c2dcd8` |
| `README` | `6a3f626147ec67b732a69beccae0aaa8` |

### 4. Updating & Recovering Data Versions

```bash
# Push new version
dvc add data/raw/ml-1m
git add data/raw/ml-1m.dvc
git commit -m "Update raw data"
dvc push
git push

# Recover previous version
git checkout <commit> -- data/raw/ml-1m.dvc
dvc checkout
```

---

## Experiment Tracking with MLflow (Epic 3 / US 3.2)

Every training is recorded in the team's **shared MLflow server on DagsHub**, so any run can be found, understood and repeated later. Everybody uses the same server and experiment, defined in `params.yaml` → `tracking`.

| | |
|---|---|
| MLflow server | [`https://dagshub.com/e26-kairi/team-B1-recsys.mlflow`](https://dagshub.com/e26-kairi/team-B1-recsys.mlflow) |
| Experiment | `recommender` |
| Access | Your own DagsHub user and token (the same ones as for DVC) |
| Code | `src/tracking/` (`ui.py`, `runs.py`, `versions.py`, `check_run.py`) |

### 1. Set Up (Once)

1. Install MLflow: `pip install -r src/tracking/requirements.txt`
2. Credentials: **nothing extra to do** if you already saved your DagsHub user and token for DVC (Data Management → step 2). They are read from `.dvc/config.local`.
   To use other credentials, set them as environment variables instead (they win over the DVC ones):
   ```bash
   # macOS/Linux
   export MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
   export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
   # Windows PowerShell
   $env:MLFLOW_TRACKING_USERNAME="<your-dagshub-username>"
   $env:MLFLOW_TRACKING_PASSWORD="<your-dagshub-token>"
   ```
   *Never write the token in `params.yaml` or in the code.*

### 2. Open the Experiment Tool (One Command)

```bash
python -m src.tracking.ui               # checks your access and opens MLflow in the browser
python -m src.tracking.ui --no-browser  # only checks access and prints the address
```

To check that logging works end to end, log a small run of the popularity baseline (tagged `purpose=tracking-check`; descriptive numbers only, the evaluation scores come from the US 2.2 training):

```bash
python -m src.tracking.check_run        # prints the run id and its address
```

| Message | Fix |
|---|---|
| `The server rejected your credentials` | Wrong user or token: repeat Data Management → step 2, or fix the environment variables. |
| `You don't have access to the team's DagsHub repository` | Ask `e26-kairi` to add you as a collaborator and accept the invitation. |
| `Can't reach ...` | Check your internet connection. |
| `No DagsHub credentials found` | Do step 1.2 above. |

### 3. Log a Training (Model Code)

Training code calls `log_training()` once per training:

```python
from src.tracking.runs import log_training, set_seed

set_seed(42)                                   # fixes the Python and NumPy random generators
model = MyModel(**settings).fit(train_df)      # needs predict_batch(df) or predict(user_id, movie_id)
run_id = log_training(
    model,
    model_name="svd",
    params=settings,                           # nested dicts are fine: {"split": {"test_size": 0.2}}
    seed=42,
    metrics={"precision_at_10": p10, "ndcg_at_10": ndcg10},
    example=test_df[["user_id", "movie_id"]],  # documents the model input in MLflow
    data_paths=["data/raw/ml-1m/ratings.dat"], # data read by the training (default: all DVC data)
)
```

What every run stores, under its own unique **run id**:

| Recorded | Where in MLflow |
|---|---|
| Settings and **random seed** | *Parameters* (`seed`, plus every setting passed in `params`) |
| Scores | *Metrics* |
| Trained model | *Logged models*; its exact address is in the tag `model_uri` (`models:/m-...`) |
| **Code version** | Tags `code.git_commit`, `code.commit_url`, `code.git_branch`; the commit is also shown in the runs table (*Version*) |
| Uncommitted code changes | Tag `code.git_dirty` (`true` means the commit alone can't repeat the run) |
| **Data version** | Tag `data.version.<path>` = DVC hash of each dataset used (e.g. `data.version.data/raw/ml-1m`) |
| Local data different from DVC | Tag `data.modified_locally` |

**Before a training you want to keep:** commit your code and run `dvc pull`. Otherwise the run is still logged, with a `WARNING` and `code.git_dirty=true` or `data.modified_locally=true`.

### 4. Find, Compare and Reproduce Runs

**Find a run by its id**
* In the browser: `https://dagshub.com/e26-kairi/team-B1-recsys.mlflow/#/experiments/0/runs/<run-id>`
* In Python:
  ```python
  from src.tracking.config import configure
  import mlflow
  configure()                                    # same server and credentials as above
  run = mlflow.get_run("<run-id>")
  print(run.data.params, run.data.metrics, run.data.tags["code.git_commit"])
  ```

**Compare runs** (e.g. baseline vs SVD)
1. Open the experiment (`python -m src.tracking.ui`).
2. Filter the table if needed, e.g. `tags.model_name = 'svd'` or `params.seed = '42'`.
3. Tick the runs → **Compare**: parameters and metrics side by side, plus charts.

**Load the model of a run**
```python
import pandas as pd
from src.tracking.runs import load_run_model
model = load_run_model("<run-id>")
model.predict(pd.DataFrame({"user_id": [1, 1], "movie_id": [1193, 661]}))   # one score per row
```
*Models are saved with cloudpickle, so loading one runs its code: only load models from the team's server.*

**Repeat a run**
1. Open the run and note `code.git_commit`, `data.version.*`, the parameters and the `seed`.
2. Get that code and its data:
   ```bash
   git checkout <code.git_commit>
   dvc checkout        # or dvc pull: restores the data versioned in that commit
   ```
3. Check that `data/raw/ml-1m.dvc` (or the dataset's `.dvc` / `dvc.lock`) shows the same hash as the run's `data.version.*` tag.
4. Train again with the same parameters and seed: the new run is logged next to the old one, ready to compare.
