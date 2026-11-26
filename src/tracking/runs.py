"""Record every training in the team's MLflow server (US3.2, task 2 / AC2).

One call per training:

    from src.tracking.runs import log_training, set_seed

    set_seed(42)                                  # same random numbers every time
    model = MyModel(...).fit(train_df)
    run_id = log_training(
        model,
        model_name="svd",
        params={"factors": 50, "epochs": 20},     # settings of this training
        seed=42,                                  # saved as the "seed" parameter
        metrics={"ndcg_at_10": 0.12},             # scores on the test set
    )

The run gets a unique id (MLflow's run id). Everything is stored under it:
parameters (including the seed), metrics and the trained model, so the run can be
found and its model loaded later with that id (see load_run_model). The version of
the code (Git commit) and of the data (DVC hash) are saved as tags (US3.2, task 3).

The model is saved as an MLflow "pyfunc" model: it receives a DataFrame with the
columns user_id and movie_id and returns one score per row. Any model object works
if it has `predict_batch(df)` (one score per row) or `predict(user_id, movie_id)`.
"""

from __future__ import annotations

import inspect
import random
import sys
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.tracking.config import TrackingSettings, configure, load_settings
from src.tracking.versions import version_tags, warnings_for

MODEL_ARTIFACT = "model"
MODEL_URI_TAG = "model_uri"
SEED_PARAM = "seed"
INPUT_COLUMNS = ["user_id", "movie_id"]


def set_seed(seed: int) -> int:
    """Fix the random generators used by the project (Python and NumPy)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """{'split': {'test_size': 0.2}} -> {'split.test_size': '0.2'} (MLflow params are flat strings)."""
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, prefix=f"{name}."))
        else:
            flat[name] = str(value)
    return flat


def _score_rows(model: Any, rows: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_batch"):
        return np.asarray(model.predict_batch(rows), dtype=float)
    return np.asarray(
        [model.predict(int(u), int(m)) for u, m in zip(rows["user_id"], rows["movie_id"])],
        dtype=float,
    )


def _pyfunc_class():
    """Built lazily so importing this module does not import mlflow."""
    import mlflow.pyfunc

    class RecommenderPyfunc(mlflow.pyfunc.PythonModel):
        """Wraps a trained recommender: (user_id, movie_id) rows -> one score per row."""

        def __init__(self, model: Any):
            self.model = model

        def predict(self, context, model_input, params=None):  # noqa: D401 (MLflow API)
            rows = pd.DataFrame(model_input).copy()
            if "movie_id" not in rows.columns:
                raise ValueError("The model input needs a 'movie_id' column.")
            if "user_id" not in rows.columns:
                rows["user_id"] = -1  # unknown user: the model answers for a new user
            return _score_rows(self.model, rows[INPUT_COLUMNS])

    return RecommenderPyfunc


def _signature():
    """Input: movie_id (required) and user_id (optional, as the API may omit it). Output: score."""
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, Schema

    inputs = Schema(
        [ColSpec("long", "user_id", required=False), ColSpec("long", "movie_id", required=True)]
    )
    outputs = Schema([ColSpec("double")])
    return ModelSignature(inputs=inputs, outputs=outputs)


def _log_model(python_model: Any, example: pd.DataFrame | None) -> str:
    """Save the model in the current run and return its address (model URI)."""
    import mlflow.pyfunc

    kwargs: dict[str, Any] = {
        "python_model": python_model,
        "code_paths": ["src"],
        "signature": _signature(),
    }
    if example is not None and not example.empty:
        kwargs["input_example"] = example[INPUT_COLUMNS].head(5).reset_index(drop=True)
    # MLflow 3 calls it `name`, MLflow 2 `artifact_path`
    if "name" in inspect.signature(mlflow.pyfunc.log_model).parameters:
        kwargs["name"] = MODEL_ARTIFACT
    else:
        kwargs["artifact_path"] = MODEL_ARTIFACT
    return mlflow.pyfunc.log_model(**kwargs).model_uri


def log_training(
    model: Any,
    *,
    model_name: str,
    params: Mapping[str, Any],
    seed: int,
    metrics: Mapping[str, float],
    example: pd.DataFrame | None = None,
    tags: Mapping[str, str] | None = None,
    data_paths: Iterable[str] | None = None,
    settings: TrackingSettings | None = None,
    repo_root: str = ".",
) -> str:
    """Save one training (params + seed, metrics, model) as a new run. Returns its id.

    `example`: a few (user_id, movie_id) rows, e.g. from the test set; they document
    the model input in MLflow. `data_paths`: the data files/folders the training read
    (e.g. ["data/processed/ratings_clean.parquet"]); their DVC version is saved with
    the run. If omitted, the version of every DVC-versioned dataset is saved.
    The Git commit of the code is always saved (see src/tracking/versions.py).
    `settings` / `repo_root`: only for tests.
    """
    import mlflow

    settings = configure(settings or load_settings())
    mlflow.set_experiment(settings.experiment)

    all_params = flatten_params(params)
    all_params[SEED_PARAM] = str(int(seed))
    versions = version_tags(data_paths, repo_root=repo_root)
    for message in warnings_for(versions):
        print(f"WARNING: {message}", file=sys.stderr)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tags({**versions, "model_name": model_name, **(tags or {})})
        mlflow.log_params(all_params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        model_uri = _log_model(_pyfunc_class()(model), example)
        # Where the model is, e.g. models:/m-... (MLflow 3). Use it to load or register it.
        mlflow.set_tag(MODEL_URI_TAG, model_uri)
        return run.info.run_id


def run_model_uri(run_id: str) -> str:
    """Address of the model saved by a run (tag set by log_training)."""
    import mlflow

    tags = mlflow.get_run(run_id).data.tags
    # runs:/<id>/model is the classic address; on DagsHub with MLflow 3 only the
    # models:/<model id> address stored in the tag can be downloaded.
    return tags.get(MODEL_URI_TAG) or f"runs:/{run_id}/{MODEL_ARTIFACT}"


def load_run_model(run_id: str, settings: TrackingSettings | None = None):
    """Load the model saved by a run, by its id (MLflow pyfunc model)."""
    import mlflow.pyfunc

    configure(settings or load_settings())
    return mlflow.pyfunc.load_model(run_model_uri(run_id))
