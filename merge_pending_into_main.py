# merge_pending_into_main.py
import os
import tempfile
import time
import json
from huggingface_hub import HfApi, hf_hub_download, CommitOperationDelete
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
from datetime import datetime, timezone

# CONFIG - set HF_TOKEN in environment or replace below (prefer env var)
HF_TOKEN = os.environ.get("HF_TOKEN")  # or set string directly (not recommended)
REPO_ID = "Isuru0x01/sinhala_stories"
REPO_TYPE = "dataset"
PENDING_DIR = "pending"   # remote repo path where small jsonl files are stored

# Validate and strip token
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable is not set. "
        "Please set it in your GitHub repository secrets or environment."
    )
HF_TOKEN = HF_TOKEN.strip()
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable is empty (only whitespace). "
        "Please set a valid Hugging Face token in your GitHub repository secrets."
    )

# Initialize API with validated token
api = HfApi(token=HF_TOKEN)

def list_repo_files():
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

def merge_and_push(pending_ds, pending_files, start_time):
    if pending_ds is None or len(pending_ds) == 0:
        print("No pending items to merge.")
        return

    print("Pending entries size:", len(pending_ds))
    
    # 1. Load the small append dataset instead of the massive main dataset
    print("Loading incremental append dataset...")
    try:
        append_ds = load_dataset(REPO_ID, data_files="data/train-append.parquet", split="train", token=HF_TOKEN)
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
    import hashlib
    import unicodedata
    import re
    from datetime import timezone
    import numpy as np

    def normalize_story_local(story):
        if not story:
            return ""
        normalized = unicodedata.normalize("NFC", story)
        normalized = re.sub(r'\.{2,}', '.', normalized)
        normalized = normalized.strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    print("Checking and populating SHA-256 hashes for pending items...")
    new_hashes = []
    # Ensure the pending dataset has 'sha256' column populated
    for row in pending_ds:
        story = row.get("story", "")
        h = row.get("sha256")
        if not h and story:
            norm_s = normalize_story_local(story)
            h = hashlib.sha256(norm_s.encode('utf-8')).hexdigest()
        if h:
            new_hashes.append(h)

    # 4. Update dataset statistics mathematically
    print("Updating dataset statistics...")
    try:
        stats_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="dataset_stats.json",
            repo_type="dataset",
            token=HF_TOKEN
        )
        with open(stats_path, "r", encoding="utf-8") as f:
            dataset_stats = json.load(f)
    except Exception:
        # Initialize with baseline stats (from your 10.9M rows)
        dataset_stats = {
            "total_stories": 10948994,
            "total_size_chars": 4515320120,  # approximate baseline character count
            "avg_len": 412.3,
            "median_len": 320.0,
            "longest_len": 24050,
            "today_count": 0,
            "week_count": 0,
            "approx_contributors": 1200
        }

    new_stories_count = len(pending_ds)
    new_lengths = [len(row.get("story", "")) for row in pending_ds if row.get("story")]
    new_total_size = sum(new_lengths)

    dataset_stats["total_stories"] += new_stories_count
    dataset_stats["total_size_chars"] += new_total_size
    if dataset_stats["total_stories"] > 0:
        dataset_stats["avg_len"] = dataset_stats["total_size_chars"] / dataset_stats["total_stories"]
    if new_lengths:
        dataset_stats["longest_len"] = max(dataset_stats["longest_len"], max(new_lengths))
    
    # Incremental update of daily/weekly counts
    dataset_stats["today_count"] += new_stories_count
    dataset_stats["week_count"] += new_stories_count
    # Incremental update of contributor count
    dataset_stats["approx_contributors"] += 1  # approximate increment per batch

    # 5. Write merged append dataset to a local parquet file
    import json
    import io
    from huggingface_hub import CommitOperationAdd
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "train-append.parquet")
        merged_append.to_parquet(parquet_path)
        
        # 6. Update hashes.txt in append mode on disk
        print("Updating hashes.txt...")
        try:
            hashes_path = hf_hub_download(
                repo_id=REPO_ID,
                filename="hashes.txt",
                repo_type="dataset",
                token=HF_TOKEN
            )
        except Exception:
            hashes_path = None
            
        if hashes_path and os.path.exists(hashes_path):
            with open(hashes_path, "a", encoding="utf-8") as f:
                for h in new_hashes:
                    f.write(h + "\n")
            hashes_upload_file = hashes_path
        else:
            hashes_upload_file = os.path.join(tmpdir, "hashes.txt")
            with open(hashes_upload_file, "w", encoding="utf-8") as f:
                for h in new_hashes:
                    f.write(h + "\n")

        duration = time.time() - start_time
        print("Uploading updated files to Hugging Face...")
        
        # Clean up processed pending files and add stats/parquet/hashes
        cleanup_operations = []
        for file_path in pending_files:
            cleanup_operations.append(CommitOperationDelete(path_in_repo=file_path))
        
        # Add parquet dataset file
        cleanup_operations.append(CommitOperationAdd(path_in_repo="data/train-append.parquet", path_or_fileobj=parquet_path))
        
        # Add hashes.txt
        cleanup_operations.append(CommitOperationAdd(path_in_repo="hashes.txt", path_or_fileobj=hashes_upload_file))

        # Add dataset_stats.json
        buf_ds_stats = io.BytesIO(json.dumps(dataset_stats, indent=2).encode("utf-8"))
        cleanup_operations.append(CommitOperationAdd(path_in_repo="dataset_stats.json", path_or_fileobj=buf_ds_stats))

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
        print("No pending files found. Nothing to do.")
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
