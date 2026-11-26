"""Code and data version of every run (US3.2 task 3, AC3).

Uses a throw-away Git repository with fake DVC pointer files: no network, no DVC cache.
"""

import subprocess

import mlflow
import pytest

from src.tracking import versions
from src.tracking.config import TrackingSettings
from src.tracking.runs import log_training

DIR_POINTER = """outs:
- md5: 82cb2289085c2178418e006f69afc645.dir
  size: 24905384
  nfiles: 4
  hash: md5
  path: ml-1m
"""

DVC_LOCK = """schema: '2.0'
stages:
  clean:
    cmd: python -m src.data.clean
    outs:
    - path: data/processed/ratings_clean.parquet
      md5: 0123456789abcdef0123456789abcdef
      size: 1000
"""


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "ml-1m.dvc").write_text(DIR_POINTER, encoding="utf-8")
    (tmp_path / "dvc.lock").write_text(DVC_LOCK, encoding="utf-8")
    (tmp_path / "train.py").write_text("print('v1')\n", encoding="utf-8")
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "remote", "add", "origin", "https://user:secret-token@github.com/org/proj.git")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "first")
    return tmp_path


def head(repo):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_code_version_is_the_commit_and_shown_by_mlflow(repo):
    tags = versions.code_version(repo)
    assert tags["code.git_commit"] == head(repo)
    assert tags["mlflow.source.git.commit"] == head(repo)  # MLflow UI "Version" column
    assert tags["code.git_branch"] == "main"
    assert tags["code.git_dirty"] == "false"


def test_repository_url_never_contains_credentials(repo):
    tags = versions.code_version(repo)
    assert tags["mlflow.source.git.repoURL"] == "https://github.com/org/proj"
    assert tags["code.commit_url"] == f"https://github.com/org/proj/commit/{head(repo)}"
    assert "secret-token" not in str(tags)


def test_uncommitted_changes_are_flagged(repo):
    (repo / "train.py").write_text("print('v2, not committed')\n", encoding="utf-8")
    tags = versions.code_version(repo)
    assert tags["code.git_dirty"] == "true"
    assert any("uncommitted" in m for m in versions.warnings_for(tags))


def test_outside_git_the_code_version_is_unknown(tmp_path):
    assert versions.code_version(tmp_path)["code.git_commit"] == versions.UNKNOWN


def test_all_dvc_versioned_data_is_found(repo):
    assert versions.tracked_data(repo) == {
        "data/raw/ml-1m": "82cb2289085c2178418e006f69afc645.dir",
        "data/processed/ratings_clean.parquet": "0123456789abcdef0123456789abcdef",
    }


def test_a_file_inside_a_versioned_folder_gets_the_folder_version(repo):
    tags = versions.data_version(["data/raw/ml-1m/ratings.dat"], repo, check_local=False)
    assert tags == {"data.version.data/raw/ml-1m": "82cb2289085c2178418e006f69afc645.dir"}


def test_pipeline_outputs_from_dvc_lock_are_versioned_too(repo):
    tags = versions.data_version(["data/processed/ratings_clean.parquet"], repo, check_local=False)
    assert tags["data.version.data/processed/ratings_clean.parquet"] == "0123456789abcdef0123456789abcdef"


def test_data_that_is_not_versioned_is_said_so(repo):
    tags = versions.data_version(["data/other.csv"], repo, check_local=False)
    assert tags["data.version.data/other.csv"] == "not versioned with DVC"


def test_absolute_paths_become_valid_tag_names(repo, tmp_path_factory):
    inside = versions.data_version([str(repo / "data" / "raw" / "ml-1m" / "ratings.dat")], repo, check_local=False)
    assert "data.version.data/raw/ml-1m" in inside
    outside = tmp_path_factory.mktemp("elsewhere") / "ratings.csv"
    (key,) = [k for k in versions.data_version([str(outside)], repo, check_local=False) if k.startswith("data.version.")]
    assert ":" not in key and "\\" not in key


def test_without_data_paths_every_dataset_is_recorded(repo):
    tags = versions.data_version(None, repo, check_local=False)
    assert {k for k in tags if k.startswith("data.version.")} == {
        "data.version.data/raw/ml-1m",
        "data.version.data/processed/ratings_clean.parquet",
    }


def test_local_data_check_uses_dvc_status(repo, monkeypatch):
    seen = {}

    def fake_status(targets, root):
        seen["targets"] = targets
        return "true"

    monkeypatch.setattr(versions, "_modified_locally", fake_status)
    tags = versions.data_version(["data/raw/ml-1m"], repo)
    assert seen["targets"] == ["data/raw/ml-1m.dvc"]
    assert tags["data.modified_locally"] == "true"
    assert any("dvc pull" in m for m in versions.warnings_for(tags))


def test_every_logged_run_has_code_and_data_version(repo, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setattr(versions, "_modified_locally", lambda targets, root: "false")
    settings = TrackingSettings(uri=(tmp_path / "mlruns").as_uri(), experiment="versions")

    class Model:
        def predict(self, user_id, movie_id):
            return 1.0

    try:
        run_id = log_training(
            Model(),
            model_name="tiny",
            params={},
            seed=1,
            metrics={},
            data_paths=["data/raw/ml-1m/ratings.dat"],
            settings=settings,
            repo_root=str(repo),
        )
        tags = mlflow.get_run(run_id).data.tags
    finally:
        mlflow.set_tracking_uri("")

    assert tags["code.git_commit"] == head(repo)
    assert tags["code.git_dirty"] == "false"
    assert tags["data.version.data/raw/ml-1m"] == "82cb2289085c2178418e006f69afc645.dir"
    assert tags["data.modified_locally"] == "false"
