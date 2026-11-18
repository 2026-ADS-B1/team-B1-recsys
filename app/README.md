# Recommendations Web Application (US 7.2)

Streamlit web interface for the movie recommendation system. It allows users to ask for recommendations directly from a graphical web page without calling the REST API manually.

## Features

- **User Identification:** Enter a specific MovieLens User ID or check **Continue as guest** to get recommendations as a new user without past ratings.
- **Mutual Exclusion Filter:** Filter recommendations by **Genre** or by **Mood**, or choose **No filter** (only one filter mode active at a time).
- **Dynamic Configuration:**
  - **Moods** are read dynamically from `params.yaml` (`serving.moods` defined in US 6.2).
  - **Genres** are populated dynamically from the movie catalogue.
- **Recommendations View:** Displays recommendations ranked by score with title and associated genres (currently displays example results according to the US 7.2 milestone specification).

## Getting Started

### 1. Install dependencies

From the root of the repository:

```bash
pip install -r app/requirements.txt
```

### 2. Run the application

```bash
streamlit run app/main.py
```

The application will start and open automatically in your browser (usually at `http://localhost:8501`).

## Testing

Run the automated Streamlit tests (headless execution using `streamlit.testing.v1.AppTest`):

```bash
pytest tests/app
```
