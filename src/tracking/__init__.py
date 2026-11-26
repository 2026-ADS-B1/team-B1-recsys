"""Experiment tracking with the team's shared MLflow server (US3.2)."""

import os

# Recent MLflow versions print an "agent hint" line on import; it is noise for the team.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
