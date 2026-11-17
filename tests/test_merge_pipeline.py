"""
tests/test_merge_pipeline.py
Unit tests for incremental dataset merging, schema harmonization,
and deterministic statistics computation.
"""

import pytest
from datetime import datetime, timezone, timedelta
from datasets import Dataset, Features, Value

from merge_pending_into_main import (
    align_dataset_schemas,
    compute_dataset_stats,
    list_pending_files,
    BASE_STORIES,
    BASE_SIZE_CHARS,
    BASE_LONGEST_LEN,
    BASE_MEDIAN_LEN,
    BASE_CONTRIBUTORS,
)


def test_compute_dataset_stats_baseline():
    """Verify compute_dataset_stats returns baseline counts when append dataset is empty/None."""
    stats = compute_dataset_stats(None)
    assert stats["total_stories"] == BASE_STORIES
    assert stats["total_size_chars"] == BASE_SIZE_CHARS
    assert stats["longest_len"] == BASE_LONGEST_LEN
    assert stats["median_len"] == BASE_MEDIAN_LEN
    assert stats["approx_contributors"] == BASE_CONTRIBUTORS
    assert stats["today_count"] == 0
    assert stats["week_count"] == 0
    assert stats["daily_counts"] == {}


def test_compute_dataset_stats_with_append_data():
    """Verify compute_dataset_stats accurately aggregates rows from the append dataset."""
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    yesterday_str = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")

    mock_append_rows = [
        {
            "story": "අ" * 100,
            "timestamp_utc": f"{today_str}T10:00:00Z",
            "contributor_session_hash": "user_a",
        },
        {
            "story": "බ" * 200,
            "timestamp_utc": f"{today_str}T12:00:00Z",
            "contributor_session_hash": "user_b",
        },
        {
            "story": "ක" * 50000,
            "timestamp_utc": f"{yesterday_str}T08:00:00Z",
            "contributor_session_hash": "user_a",  # Duplicate contributor
        },
    ]

    mock_ds = Dataset.from_list(mock_append_rows)
    stats = compute_dataset_stats(mock_ds)

    # Row counts
    assert stats["total_stories"] == BASE_STORIES + 3
    # Total character size
    assert stats["total_size_chars"] == BASE_SIZE_CHARS + 100 + 200 + 50000
    # Longest length
    assert stats["longest_len"] == 50000
    # Unique contributors (user_a, user_b -> 2 unique)
    assert stats["approx_contributors"] == BASE_CONTRIBUTORS + 2
    # Daily counts
    assert stats["daily_counts"][today_str] == 2
    assert stats["daily_counts"][yesterday_str] == 1
    assert stats["today_count"] == 2
    assert stats["week_count"] == 3


def test_align_dataset_schemas():
    """Verify align_dataset_schemas harmonizes differing columns across datasets."""
    ds1 = Dataset.from_dict({
        "story": ["කතාව 1"],
        "language": ["si"],
    })
    ds2 = Dataset.from_dict({
        "story": ["කතාව 2"],
        "adult": [False],
        "language": ["si"],
    })

    aligned = align_dataset_schemas([ds1, ds2])
    assert len(aligned) == 2

    # Both datasets should now have identical features
    assert set(aligned[0].column_names) == set(aligned[1].column_names)
    assert "adult" in aligned[0].column_names
    assert "adult" in aligned[1].column_names

    # Missing values should be None
    assert aligned[0][0]["adult"] is None
    assert aligned[1][0]["adult"] is False


def test_list_pending_files():
    """Verify list_pending_files correctly filters files inside the pending/ directory."""
    sample_files = [
        "README.md",
        "data/train-00000-of-00010.parquet",
        "data/train-append.parquet",
        "hashes.txt",
        "dataset_stats.json",
        "pending/entry_20261115T145352Z_ea811309_1ba12d.jsonl",
        "pending/entry_20261116T173050Z_647dc44e_f395a3.jsonl",
        "pending/not_a_jsonl.txt",
        "other_dir/entry_test.jsonl",
    ]

    pending = list_pending_files(sample_files)
    assert len(pending) == 2
    assert "pending/entry_20261115T145352Z_ea811309_1ba12d.jsonl" in pending
    assert "pending/entry_20261116T173050Z_647dc44e_f395a3.jsonl" in pending
    assert "pending/not_a_jsonl.txt" not in pending

