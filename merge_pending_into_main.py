# merge_pending_into_main.py
import os
import tempfile
import time
import json
from huggingface_hub import HfApi, hf_hub_download, CommitOperationDelete
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
from datetime import datetime, timezone, timedelta

from validator import load_config, normalize_text, compute_sha256, BloomFilter, build_bloom_filter

# Load settings from config.yaml with default fallbacks
CONFIG = load_config()
PROJECT_CFG = CONFIG.get("project", {})
STORAGE_CFG = CONFIG.get("storage", {})

# Exact baseline counts from original 10 shards (train-00000 to train-00009)
BASE_STORIES = 10949004
BASE_SIZE_CHARS = 1231746068
BASE_LONGEST_LEN = 49974
BASE_MEDIAN_LEN = 94.0
BASE_CONTRIBUTORS = 1200

# CONFIG - set HF_TOKEN in environment or replace below (prefer env var)
HF_TOKEN = os.environ.get("HF_TOKEN")  # or set string directly (not recommended)
REPO_ID = PROJECT_CFG.get("hf_repo_id", "Isuru0x01/sinhala_stories")
REPO_TYPE = "dataset"
PENDING_DIR = STORAGE_CFG.get("pending_dir", "pending")   # remote repo path where small jsonl files are stored
APPEND_PARQUET_PATH = STORAGE_CFG.get("append_parquet_path", "data/train-append.parquet")
HASHES_PATH = STORAGE_CFG.get("hashes_path", "hashes.txt")
BLOOM_PATH = STORAGE_CFG.get("bloom_path", "hashes.bloom")
STATS_PATH = STORAGE_CFG.get("stats_path", "dataset_stats.json")

def get_hf_api() -> HfApi:
    """Returns initialized HfApi instance with validated token."""
    token = os.environ.get("HF_TOKEN")
    if not token or not token.strip():
        raise ValueError(
            "HF_TOKEN environment variable is not set. "
            "Please set it in your GitHub repository secrets or environment."
        )
    return HfApi(token=token.strip())

def list_repo_files():
    api = get_hf_api()
    return api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

def list_pending_files(files):
    return [f for f in files if f.startswith(PENDING_DIR + "/") and f.endswith(".jsonl")]

def download_file(filename, outdir):
    os.makedirs(outdir, exist_ok=True)
    # hf_hub_download writes to cache, so returns path to cached file
    local = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type=REPO_TYPE, token=HF_TOKEN)
    target = os.path.join(outdir, os.path.basename(filename))
    with open(local, "rb") as r, open(target, "wb") as w:
        w.write(r.read())
    return target

def align_dataset_schemas(datasets):
    if not datasets:
        return []
    
    # 1. Collect union of all features
    all_features = {}
    for ds in datasets:
        for col_name, feature in ds.features.items():
            if col_name not in all_features:
                all_features[col_name] = feature
                
    # Define the target features schema
    target_features = Features(all_features)
    
    aligned_datasets = []
    for ds in datasets:
        missing_cols = set(all_features.keys()) - set(ds.column_names)
        if not missing_cols:
            # Just cast to ensure the exact same feature types (e.g. string, float, etc.)
            aligned_datasets.append(ds.cast(target_features))
            continue
            
        def add_missing_keys(batch):
            # batch is a dict mapping column name -> list of values
            first_key = list(batch.keys())[0]
            batch_size = len(batch[first_key])
            for col in missing_cols:
                batch[col] = [None] * batch_size
            return batch
            
        aligned_ds = ds.map(
            add_missing_keys,
            batched=True,
            features=target_features,
            desc="Aligning schemas",
            keep_in_memory=True
        )
        aligned_datasets.append(aligned_ds)
        
    return aligned_datasets

def load_pending_datasets(local_files):
    datasets = []
    for f in local_files:
        # each file is jsonl - create a Dataset from it
        ds = Dataset.from_json(f)
        datasets.append(ds)
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    
    # Align schemas of all pending datasets
    aligned = align_dataset_schemas(datasets)
    return concatenate_datasets(aligned, axis=0)

def compute_dataset_stats(merged_append):
    """
    Computes exact dataset statistics deterministically from the base dataset constants
    and the full merged_append dataset.
    """
    append_len = len(merged_append) if merged_append is not None else 0
    append_size = sum(len(row.get("story", "") or "") for row in merged_append) if merged_append is not None else 0
    append_longest = max((len(row.get("story", "") or "") for row in merged_append), default=0) if merged_append is not None else 0
    append_contribs = len(set(row.get("contributor_session_hash") for row in merged_append if row.get("contributor_session_hash"))) if merged_append is not None else 0

    now_utc = datetime.now(timezone.utc)
    today_key = now_utc.strftime("%Y-%m-%d")
    rolling_7d_keys = [(now_utc - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

    daily_counts = {}
    if merged_append is not None:
        for row in merged_append:
            ts = row.get("timestamp_utc") or ""
            day_key = None
            if len(ts) >= 10 and ts[4] == '-' and ts[7] == '-':
                day_key = ts[:10]
            elif len(ts) >= 8 and ts[:8].isdigit():
                day_key = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
            if day_key:
                daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

    total_stories = BASE_STORIES + append_len
    total_size_chars = BASE_SIZE_CHARS + append_size

    return {
        "total_stories": total_stories,
        "total_size_chars": total_size_chars,
        "avg_len": total_size_chars / total_stories if total_stories > 0 else 0.0,
        "median_len": BASE_MEDIAN_LEN,
        "longest_len": max(BASE_LONGEST_LEN, append_longest),
        "today_count": daily_counts.get(today_key, 0),
        "week_count": sum(daily_counts.get(k, 0) for k in rolling_7d_keys),
        "daily_counts": daily_counts,
        "approx_contributors": BASE_CONTRIBUTORS + append_contribs,
    }

def sync_stats_if_needed():
    """Checks if dataset_stats.json and hashes.bloom on Hugging Face need synchronization."""
    print("Checking if dataset_stats.json and hashes.bloom need synchronization...")
    try:
        stats_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=STATS_PATH,
            repo_type="dataset",
            token=HF_TOKEN,
            force_download=True
        )
        with open(stats_path, "r", encoding="utf-8") as f:
            current_stats = json.load(f)
    except Exception:
        current_stats = {}

    try:
        append_ds = load_dataset(REPO_ID, data_files=APPEND_PARQUET_PATH, split="train", token=HF_TOKEN)
    except Exception:
        append_ds = None

    fresh_stats = compute_dataset_stats(append_ds)
    stats_out_of_sync = not (
        current_stats.get("total_stories") == fresh_stats["total_stories"]
        and current_stats.get("total_size_chars") == fresh_stats["total_size_chars"]
        and current_stats.get("longest_len") == fresh_stats["longest_len"]
    )

    # Check if binary Bloom Filter exists
    bloom_missing = False
    try:
        hf_hub_download(repo_id=REPO_ID, filename=BLOOM_PATH, repo_type="dataset", token=HF_TOKEN)
    except Exception:
        bloom_missing = True

    if not stats_out_of_sync and not bloom_missing:
        print(f"Dataset stats ({fresh_stats['total_stories']} stories) and Bloom Filter are already synchronized.")
        return

    operations = []
    import io
    from huggingface_hub import CommitOperationAdd

    if stats_out_of_sync:
        print(f"Stats out of sync! Current in repo: {current_stats.get('total_stories')}, Expected: {fresh_stats['total_stories']}. Updating...")
        buf_ds_stats = io.BytesIO(json.dumps(fresh_stats, indent=2).encode("utf-8"))
        operations.append(CommitOperationAdd(path_in_repo=STATS_PATH, path_or_fileobj=buf_ds_stats))

    if bloom_missing:
        print("hashes.bloom is missing in repository. Compiling from hashes.txt...")
        try:
            h_path = hf_hub_download(repo_id=REPO_ID, filename=HASHES_PATH, repo_type="dataset", token=HF_TOKEN)
            with open(h_path, "r", encoding="utf-8") as f:
                all_h = [l.strip() for l in f if l.strip()]
        except Exception:
            all_h = []
        bf = build_bloom_filter(all_h, capacity=max(len(all_h) * 2, 100000), error_rate=0.001)
        buf_bloom = io.BytesIO(bf.to_bytes())
        operations.append(CommitOperationAdd(path_in_repo=BLOOM_PATH, path_or_fileobj=buf_bloom))
        print(f"Compiled binary Bloom Filter ({len(all_h)} fingerprints, {len(bf.to_bytes())} bytes).")

    if operations:
        api = get_hf_api()
        api.create_commit(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            operations=operations,
            commit_message=f"Sync dataset metadata (stats: {fresh_stats['total_stories']} stories, Bloom Filter compiled)"
        )
        print("Successfully synced metadata to Hugging Face repository.")

def merge_and_push(pending_ds, pending_files, start_time):
    if pending_ds is None or len(pending_ds) == 0:
        print("No pending items to merge.")
        return

    print("Pending entries size:", len(pending_ds))
    
    # 1. Load the small append dataset instead of the massive main dataset
    print("Loading incremental append dataset...")
    try:
        append_ds = load_dataset(REPO_ID, data_files=APPEND_PARQUET_PATH, split="train", token=HF_TOKEN)
        print("Existing append dataset loaded. Size:", len(append_ds))
    except Exception:
        print("No existing append dataset found. Starting fresh.")
        append_ds = None

    # 2. Align schemas and concatenate pending stories to the append dataset
    if append_ds is not None:
        print("Aligning schemas...")
        aligned = align_dataset_schemas([append_ds, pending_ds])
        merged_append = concatenate_datasets(aligned)
    else:
        merged_append = pending_ds
        
    print("Total append dataset size:", len(merged_append))

    # 3. Add/Update SHA-256 hashes in pending dataset
    from datetime import timezone
    import numpy as np

    print("Checking and populating SHA-256 hashes for pending items...")
    new_hashes = []
    # Ensure the pending dataset has 'sha256' column populated
    for row in pending_ds:
        story = row.get("story", "")
        h = row.get("sha256")
        if not h and story:
            norm_s = normalize_text(story)
            h = compute_sha256(norm_s)
        if h:
            new_hashes.append(h)

    # 4. Update dataset statistics mathematically
    print("Computing exact dataset statistics...")
    dataset_stats = compute_dataset_stats(merged_append)

    # 5. Write merged append dataset to a local parquet file
    import json
    import io
    from huggingface_hub import CommitOperationAdd
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "train-append.parquet")
        merged_append.to_parquet(parquet_path)
        
        # 6. Update hashes.txt and generate binary Bloom Filter
        print("Updating hashes.txt and generating hashes.bloom...")
        try:
            hashes_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=HASHES_PATH,
                repo_type="dataset",
                token=HF_TOKEN,
                force_download=True
            )
        except Exception:
            hashes_path = None
            
        if hashes_path and os.path.exists(hashes_path):
            with open(hashes_path, "a", encoding="utf-8") as f:
                for h in new_hashes:
                    f.write(h + "\n")
            hashes_upload_file = hashes_path
        else:
            hashes_upload_file = os.path.join(tmpdir, os.path.basename(HASHES_PATH))
            with open(hashes_upload_file, "w", encoding="utf-8") as f:
                for h in new_hashes:
                    f.write(h + "\n")

        with open(hashes_upload_file, "r", encoding="utf-8") as f:
            all_hashes = [line.strip() for line in f if line.strip()]

        bloom = build_bloom_filter(all_hashes, capacity=max(len(all_hashes) * 2, 100000), error_rate=0.001)
        bloom_upload_file = os.path.join(tmpdir, os.path.basename(BLOOM_PATH))
        bloom.save(bloom_upload_file)
        print(f"Bloom Filter generated: {os.path.getsize(bloom_upload_file)} bytes for {len(all_hashes)} fingerprints.")

        duration = time.time() - start_time
        print("Uploading updated files to Hugging Face...")
        
        # Clean up processed pending files and add stats/parquet/hashes/bloom
        cleanup_operations = []
        for file_path in pending_files:
            cleanup_operations.append(CommitOperationDelete(path_in_repo=file_path))
        
        # Add parquet dataset file
        cleanup_operations.append(CommitOperationAdd(path_in_repo=APPEND_PARQUET_PATH, path_or_fileobj=parquet_path))
        
        # Add hashes.txt
        cleanup_operations.append(CommitOperationAdd(path_in_repo=HASHES_PATH, path_or_fileobj=hashes_upload_file))

        # Add hashes.bloom
        cleanup_operations.append(CommitOperationAdd(path_in_repo=BLOOM_PATH, path_or_fileobj=bloom_upload_file))

        # Add dataset_stats.json
        buf_ds_stats = io.BytesIO(json.dumps(dataset_stats, indent=2).encode("utf-8"))
        cleanup_operations.append(CommitOperationAdd(path_in_repo=STATS_PATH, path_or_fileobj=buf_ds_stats))

        # Add merge_stats.json
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stats = {
            "last_merge_timestamp": ts,
            "duration_seconds": duration,
            "merged_entries_count": new_stories_count
        }
        buf_stats = io.BytesIO(json.dumps(stats, indent=2).encode("utf-8"))
        cleanup_operations.append(CommitOperationAdd(path_in_repo="merge_stats.json", path_or_fileobj=buf_stats))
        
        if cleanup_operations:
            api = get_hf_api()
            api.create_commit(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                operations=cleanup_operations,
                commit_message=f"Merge pending submissions ({ts}) — merged {new_stories_count} entries [optimized]"
            )
            print(f"Cleaned up pending files and updated metadata files. Duration: {duration:.2f}s")

def main():
    import time
    start_time = time.time()
    files = list_repo_files()
    pending = list_pending_files(files)
    if not pending:
        print("No pending files found.")
        sync_stats_if_needed()
        return

    print("Found pending files:", pending)
    tmp = tempfile.mkdtemp(prefix="hf_merge_")
    local_pending = []
    for fn in pending:
        print("Downloading", fn)
        local = download_file(fn, tmp)
        local_pending.append(local)

    pending_ds = load_pending_datasets(local_pending)
    if pending_ds is None:
        print("No pending dataset created. Exiting.")
        return

    # Do the merge and push
    merge_and_push(pending_ds, pending, start_time)

if __name__ == "__main__":
    main()
