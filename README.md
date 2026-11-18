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
