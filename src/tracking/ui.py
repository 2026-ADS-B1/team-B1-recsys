"""Open the team's experiment tool with one command (US3.2, task 1).

From the repository root:

    python -m src.tracking.ui              # check access and open MLflow in the browser
    python -m src.tracking.ui --no-browser # only check access and print the address

It uses the shared settings of params.yaml (section `tracking`), so every team
member lands on the same server and experiment. If something is wrong it says
what and how to fix it, instead of a long traceback.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser

from src.tracking.config import TrackingError, TrackingSettings, configure, load_settings


def experiment_url(settings: TrackingSettings, experiment_id: str | None) -> str:
    """Address of the MLflow page to open (the experiment if it exists)."""
    if experiment_id is None:
        return f"{settings.uri}/#/experiments"
    return f"{settings.uri}/#/experiments/{experiment_id}"


def _explain(exc: Exception, settings: TrackingSettings) -> str:
    text = str(exc)
    if any(code in text for code in ("401", "Unauthorized", "Bad credentials")):
        return (
            "The server rejected your credentials. Check your DagsHub user and token "
            "(README -> Data, step 3) or MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD."
        )
    if any(code in text for code in ("403", "Forbidden", "404", "Not Found")):
        return (
            "You don't have access to the team's DagsHub repository. Ask its owner to add "
            "you as a collaborator and accept the invitation."
        )
    if any(word in text for word in ("Max retries", "Connection", "timed out", "resolve")):
        return f"Can't reach {settings.uri}. Check your internet connection."
    return f"Can't use the MLflow server at {settings.uri}: {text[:300]}"


def check_connection(settings: TrackingSettings) -> str | None:
    """Contact the server. Returns the experiment id (None if it doesn't exist yet)."""
    import mlflow

    client = mlflow.MlflowClient()
    try:
        client.search_experiments(max_results=1)
        experiment = client.get_experiment_by_name(settings.experiment)
    except Exception as exc:  # network, auth, server errors...
        raise TrackingError(_explain(exc, settings)) from exc
    return experiment.experiment_id if experiment else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.tracking.ui",
        description="Check access to the team's MLflow server and open it in the browser.",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="only check access and print the address"
    )
    args = parser.parse_args(argv)

    try:
        settings = configure(load_settings())
        experiment_id = check_connection(settings)
    except TrackingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    url = experiment_url(settings, experiment_id)
    print(f"MLflow server : {settings.uri}")
    if experiment_id is None:
        print(
            f"Experiment    : '{settings.experiment}' (no runs yet; it is created by the first training)"
        )
    else:
        print(f"Experiment    : '{settings.experiment}' (id {experiment_id})")
    print(f"Open          : {url}")
    if not args.no_browser:
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
