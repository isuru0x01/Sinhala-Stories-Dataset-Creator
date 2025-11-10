# streamlit_safe_append.py
import streamlit as st
from huggingface_hub import HfApi, CommitOperationAdd
from datetime import datetime
import io, json, re

# CONFIG
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
DATASET_REPO = "Isuru0x01/sinhala_stories"
PENDING_DIR = "pending"              # folder in repo where we append small jsonl files
MIN_STORY_LENGTH = 50
MAX_STORY_LENGTH = 50000

st.set_page_config(page_title="Sinhala Story Submission (Safe Append)", layout="wide")
st.title("📚 Sinhala Story Submission — Safe Append")

def validate_story(story: str):
    errors = []
    s = story.strip()
    if not s:
        errors.append("❌ Story cannot be empty")
    elif len(s) < MIN_STORY_LENGTH:
        errors.append(f"❌ Must be at least {MIN_STORY_LENGTH} characters")
    elif len(s) > MAX_STORY_LENGTH:
        errors.append(f"❌ Exceeds maximum {MAX_STORY_LENGTH} characters")
    if not any(0x0D80 <= ord(c) <= 0x0DFF for c in s):
        errors.append("⚠️ No Sinhala characters detected")
    return errors

def upload_jsonl_to_pending(story: str):
    try:
        api = HfApi(token=HUGGINGFACE_TOKEN)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        payload = {
            "story": story.strip(),
            "timestamp_utc": timestamp,
            "status": "pending"  # Add status flag for tracking
        }
    except Exception as e:
        st.error(f"Failed to initialize Hugging Face API: {str(e)}")
        raise
    buf = io.BytesIO(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
    filename = f"{PENDING_DIR}/entry_{timestamp}.jsonl"
    api.create_commit(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=filename, path_or_fileobj=buf)],
        commit_message=f"Add pending submission {timestamp}"
    )
    return filename

story = st.text_area("Write your story (Sinhala)", height=300, placeholder="ඔබේ කතාව මෙහි ලියන්න...")
if st.button("Submit"):
    errors = validate_story(story)
    if errors:
        for e in errors:
            st.error(e)
    else:
        try:
            filename = upload_jsonl_to_pending(story)
            st.success("✅ Submitted safely — stored in pending/ on Hugging Face.")
            st.info(f"File created: `{filename}`")
            st.write("A separate merge process (local or CI) must run to merge pending/ → main dataset.")
        except Exception as e:
            st.error(f"Upload failed: {e}")
