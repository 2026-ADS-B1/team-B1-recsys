"""Unit tests for batch recommendation execution, parquet export, and logging.

Validates Acceptance Criteria for US 2.1:
- AC1: Job completes successfully and produces recommendations.
- AC5: Parquet export format and structure for downstream services.
- AC6: Execution status and metadata logging.
"""
from pathlib import Path
import pytest
import pandas as pd

from src.models.baseline import export_recommendations_to_parquet
from src.models.batch_recommend import run_batch_job, setup_logger


@pytest.fixture
def sample_ratings():
    """Create sample ratings dataset."""
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3, 3, 4],
        "movie_id": [10, 20, 30, 10, 40, 20, 30, 40],
        "rating": [5, 4, 5, 3, 4, 5, 2, 4],
        "timestamp": [100, 101, 102, 103, 104, 105, 106, 107],
    })


@pytest.fixture
def sample_movies():
    """Create sample movies metadata."""
    return pd.DataFrame({
        "movie_id": [10, 20, 30, 40],
        "title": ["Movie 10", "Movie 20", "Movie 30", "Movie 40"],
        "genres": ["Action", "Comedy", "Drama", "Sci-Fi"],
    })


def test_export_recommendations_to_parquet_success(tmp_path):
    """AC5: Test exporting recommendations to Parquet file."""
    recs_df = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "rank": [1, 2, 1, 2],
        "movie_id": [10, 20, 30, 40],
        "score": [4.8, 4.2, 4.5, 3.9],
        "title": ["M10", "M20", "M30", "M40"],
    })

    target_file = tmp_path / "output" / "recommendations.parquet"
    result_path = export_recommendations_to_parquet(recs_df, target_file)

    assert result_path.exists()
    assert result_path.suffix == ".parquet"

    # Read back and verify
    loaded = pd.read_parquet(result_path)
    assert len(loaded) == 4
    assert list(loaded.columns) == ["user_id", "rank", "movie_id", "score", "title"]
    assert loaded["user_id"].tolist() == [1, 1, 2, 2]


def test_export_recommendations_to_parquet_empty_raises(tmp_path):
    """Test validation when exporting empty or None DataFrame."""
    target_file = tmp_path / "empty.parquet"
    with pytest.raises(ValueError, match="Cannot export empty or None"):
        export_recommendations_to_parquet(pd.DataFrame(), target_file)


def test_export_recommendations_to_parquet_missing_cols(tmp_path):
    """Test validation when required columns are missing."""
    target_file = tmp_path / "invalid.parquet"
    invalid_df = pd.DataFrame({"some_col": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing required columns"):
        export_recommendations_to_parquet(invalid_df, target_file)


def test_run_batch_job_success_e2e(sample_ratings, sample_movies, tmp_path):
    """AC1, AC5, AC6: End-to-end batch execution with Parquet export and status metadata."""
    output_parquet = tmp_path / "batch_out" / "recs.parquet"
    log_file = tmp_path / "logs" / "batch.log"

    metadata = run_batch_job(
        ratings_df=sample_ratings,
        movies_df=sample_movies,
        output_path=output_parquet,
        top_n=2,
        min_ratings=1,
        log_file=log_file,
    )

    # AC6: Verify execution status and metadata
    assert metadata["status"] == "SUCCESS"
    assert metadata["error"] is None
    assert metadata["users_count"] == 4
    assert metadata["recommendations_count"] > 0
    assert metadata["duration_seconds"] >= 0.0
    assert metadata["output_size_bytes"] > 0
    assert metadata["start_time"] is not None
    assert metadata["end_time"] is not None

    # AC5: Verify output file exists and can be loaded
    assert output_parquet.exists()
    recs = pd.read_parquet(output_parquet)
    assert "user_id" in recs.columns
    assert "rank" in recs.columns
    assert "movie_id" in recs.columns
    assert "score" in recs.columns

    # Verify log file was written (AC6)
    assert log_file.exists()
    log_content = log_file.read_text(encoding="utf-8")
    assert "SUCCESS" in log_content
    assert "recsys.batch" in log_content


def test_run_batch_job_specific_users(sample_ratings, sample_movies, tmp_path):
    """Test batch job filtered to specific target users."""
    output_parquet = tmp_path / "recs_sub.parquet"

    metadata = run_batch_job(
        ratings_df=sample_ratings,
        movies_df=sample_movies,
        user_ids=[1, 3],
        output_path=output_parquet,
        top_n=2,
        min_ratings=1,
    )

    assert metadata["status"] == "SUCCESS"
    assert metadata["users_count"] == 2

    recs = pd.read_parquet(output_parquet)
    assert set(recs["user_id"].unique()) == {1, 3}


def test_run_batch_job_failure_logs_and_raises(tmp_path):
    """AC6: Test that failure sets status FAILED and raises exception."""
    output_parquet = tmp_path / "should_not_exist.parquet"

    with pytest.raises(ValueError, match="ratings_df cannot be empty"):
        run_batch_job(
            ratings_df=pd.DataFrame(),
            output_path=output_parquet,
        )

    assert not output_parquet.exists()

