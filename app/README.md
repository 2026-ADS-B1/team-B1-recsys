# Recommendations Web Application (US 7.2)

Streamlit web interface for the movie recommendation system. It allows users to ask for recommendations directly from a graphical web page without calling the REST API manually.

## Features

- **User Identification:** Enter a specific MovieLens User ID or check **Continue as guest** to get recommendations as a new user without past ratings.
- **Mutual Exclusion Filter:** Filter recommendations by **Genre** or by **Mood**, or choose **No filter** (only one filter mode active at a time).
- **Dynamic Configuration:**
  - **Service Address:** Configured in `params.yaml` under `app.service_url` (default: `http://127.0.0.1:8000`), or overridden via the `SERVICE_URL` environment variable.
  - **Moods:** Read dynamically from `params.yaml` (`serving.moods` defined in US 6.2).
  - **Genres:** Loaded from the recommendation service (`GET /genres`) with fallback to the local movie catalogue.
- **Real Service Integration & Waiting Indicator:**
  - Queries `POST /recommend` with user ID (or none for guest) and the selected genre or mood.
  - Displays a waiting indicator (`st.spinner`) while waiting for the service response.
  - Presents recommendations in the exact order received, displaying rank, title, genres, and popularity score.
- **User-Friendly Error Handling (AC4 & AC5):**
  - **User not found (404):** Displays a friendly warning explaining that the User ID is not registered, and offers an instant button to continue as a guest.
  - **Service Unavailable (ConnectError):** Displays a friendly alert explaining that the backend service is not running and shows how to start it.
  - **Service Timeout (TimeoutException):** Displays a friendly alert explaining that the service is taking longer than expected and to try again shortly.
- **Decoupled Architecture:** No ML models or database dependencies are loaded directly in the web application.

---

## How to Start the App with the Service

The web application acts as a client to the recommendation API. To run the full system, you need **two terminal sessions**:

### 1. Terminal 1: Start the Recommendation Service (FastAPI)

From the root of the repository:

```bash
uvicorn api.main:app --reload
```

The service starts at `http://127.0.0.1:8000`. You can verify it is healthy by checking:
* Health check: http://127.0.0.1:8000/health
* API Documentation: http://127.0.0.1:8000/docs

### 2. Terminal 2: Start the Web Application (Streamlit)

From the root of the repository in a second terminal:

```bash
streamlit run app/main.py
```

Streamlit will automatically open your browser at `http://localhost:8501`.

*(If you have multiple Python environments on Windows, you can explicitly use your environment's python, e.g.: `& "C:\Users\<user>\anaconda3\python.exe" -m streamlit run app/main.py`)*.

### 3. Custom Service Address (Optional)

If your API is running on a different port or server, set the `SERVICE_URL` environment variable:

```bash
SERVICE_URL="http://127.0.0.1:8000" streamlit run app/main.py
```

Or update `app.service_url` in `params.yaml`.

---

## Testing

Run the automated test suite with pytest (uses headless execution and fake service mocks):

```bash
pytest tests/app
```

To run all repository tests (API, app, and models):

```bash
pytest
```
