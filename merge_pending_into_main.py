# merge_pending_into_main.py
import os
import tempfile
from huggingface_hub import HfApi, hf_hub_download, CommitOperationDelete
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
from datetime import datetime

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
    # Load current main dataset fully (this will download shards - ensure you have disk/memory)
    print("Loading main dataset (this may be large)...")
    main_ds = load_dataset(REPO_ID, split="train", token=HF_TOKEN)
    print("Main dataset loaded. Size:", len(main_ds))
    if pending_ds is None or len(pending_ds) == 0:
        print("No pending items to merge.")
        return

    print("Pending entries size:", len(pending_ds))
    
    # Align schemas of main_ds and pending_ds
    print("Aligning schemas between main and pending datasets...")
    aligned = align_dataset_schemas([main_ds, pending_ds])
    
    # Concatenate
    merged = concatenate_datasets(aligned)
    print("Merged dataset size:", len(merged))

    # Optionally deduplicate here if you want (not included)

    # Push merged dataset as a new commit (this will replace dataset files on main with new shards)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    commit_msg = f"Merge pending submissions ({ts}) — merged {len(pending_ds)} entries"
    print("Pushing merged dataset to Hugging Face (this may take a while)...")
    merged.push_to_hub(REPO_ID, token=HF_TOKEN, commit_message=commit_msg)
    print("Push complete.")
    
    duration = time.time() - start_time
    
    # Clean up processed pending files and add stats
    print("Cleaning up processed pending files and writing merge_stats.json...")
    cleanup_operations = []
    for file_path in pending_files:
        cleanup_operations.append(CommitOperationDelete(path_in_repo=file_path))
    
    # Create merge stats
    stats = {
        "last_merge_timestamp": ts,
        "duration_seconds": duration,
        "merged_entries_count": len(pending_ds)
    }
    from huggingface_hub import CommitOperationAdd
    import io
    buf_stats = io.BytesIO(json.dumps(stats, indent=2).encode("utf-8"))
    cleanup_operations.append(CommitOperationAdd(path_in_repo="merge_stats.json", path_or_fileobj=buf_stats))
    
    if cleanup_operations:
        api.create_commit(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            operations=cleanup_operations,
            commit_message=f"Cleanup processed pending files & update merge stats ({ts})"
        )
        print(f"Cleaned up pending files and updated merge_stats.json. Duration: {duration:.2f}s")

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
