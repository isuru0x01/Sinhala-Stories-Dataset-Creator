# merge_pending_into_main.py
import os
import glob
import json
import tempfile
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, Dataset, concatenate_datasets
from datetime import datetime

# CONFIG - set HF_TOKEN in environment or replace below (prefer env var)
HF_TOKEN = os.environ.get("HF_TOKEN")  # or set string directly (not recommended)
REPO_ID = "Isuru0x01/sinhala_stories"
REPO_TYPE = "dataset"
PENDING_DIR = "pending"   # remote repo path where small jsonl files are stored

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
    return concatenate_datasets(datasets, axis=0)

def merge_and_push(pending_ds, pending_files):
    # Load current main dataset fully (this will download shards - ensure you have disk/memory)
    print("Loading main dataset (this may be large)...")
    main_ds = load_dataset(REPO_ID, split="train", use_auth_token=HF_TOKEN)
    print("Main dataset loaded. Size:", len(main_ds))
    if pending_ds is None or len(pending_ds) == 0:
        print("No pending items to merge.")
        return

    print("Pending entries size:", len(pending_ds))
    # Concatenate
    merged = concatenate_datasets([main_ds, pending_ds])
    print("Merged dataset size:", len(merged))

    # Optionally deduplicate here if you want (not included)

    # Push merged dataset as a new commit (this will replace dataset files on main with new shards)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    commit_msg = f"Merge pending submissions ({ts}) — merged {len(pending_ds)} entries"
    print("Pushing merged dataset to Hugging Face (this may take a while)...")
    merged.push_to_hub(REPO_ID, token=HF_TOKEN, commit_message=commit_msg)
    print("Push complete.")
    
    # Clean up processed pending files
    print("Cleaning up processed pending files...")
    cleanup_operations = []
    for file_path in pending:
        cleanup_operations.append({
            "operation": "delete",
            "path": file_path
        })
    
    if cleanup_operations:
        api.create_commit(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            operations=cleanup_operations,
            commit_message=f"Cleanup processed pending files ({ts})"
        )
        print(f"Cleaned up {len(cleanup_operations)} pending files.")

def main():
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
    merge_and_push(pending_ds, pending)

if __name__ == "__main__":
    main()
