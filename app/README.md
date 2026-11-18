# Recommendations Web Application (US 7.2)

Streamlit web interface for the movie recommendation system. It allows users to ask for recommendations directly from a graphical web page without calling the REST API manually.

## Features

- **User Identification:** Enter a specific MovieLens User ID or check **Continue as guest** to get recommendations as a new user without past ratings.
- **Mutual Exclusion Filter:** Filter recommendations by **Genre** or by **Mood**, or choose **No filter** (only one filter mode active at a time).
- **Dynamic Configuration:**
  - **Service Address:** Configured in `params.yaml` under `app.service_url` (default: `http://127.0.0.1:8000`), or overridden via the `SERVICE_URL` environment variable.
  - **Moods:** Read dynamically from `params.yaml` (`serving.moods` defined in US 6.2).
  - **Genres:** Loaded from the recommendation service (`GET /genres`) with fallback to the local movie catalogue.
- **Real Service Integration:**
  - Queries `POST /recommend` with user ID (or none for guest) and the selected genre or mood.
  - Displays a waiting indicator (`st.spinner`) while waiting for the service response.
  - Presents recommendations in the exact order received, displaying rank, title, genres, and popularity score.
  - Displays user-friendly error alerts when a user is not found (404) or the service is unreachable.
- **Decoupled Architecture:** No ML models or database dependencies are loaded directly in the web application.

## Getting Started

### 1. Install dependencies

From the root of the repository:

```bash
pip install -r app/requirements.txt
```

### 2. Start the recommendation API service

In a terminal:

```bash
uvicorn api.main:app --reload
```

### 3. Run the web application

In another terminal:

```bash
streamlit run app/main.py
```

The web application will open automatically at `http://localhost:8501`.

To use a custom service address:

```bash
SERVICE_URL="http://my-api-server:8000" streamlit run app/main.py
```

## Testing

Run the automated Streamlit tests (headless execution using `streamlit.testing.v1.AppTest` and fake service mocks):

```bash
pytest tests/app
```
