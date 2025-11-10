# streamlit_safe_append.py
import streamlit as st
from huggingface_hub import HfApi, CommitOperationAdd
from datetime import datetime, timezone
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

def get_api():
    """Get initialized HfApi instance."""
    return HfApi(token=HUGGINGFACE_TOKEN)

def get_pending_stories_count():
    """Get the number of pending stories in the repository."""
    try:
        api = get_api()
        files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
        pending_files = [f for f in files if f.startswith(PENDING_DIR + "/") and f.endswith(".jsonl")]
        return len(pending_files)
    except Exception as e:
        # Don't show error here - return None and handle in UI
        return None

def get_last_merge_timestamp():
    """
    Get the timestamp of the last successful merge.
    Attempts to retrieve commit history from the Hugging Face repository.
    Returns None if commit history is not available.
    """
    try:
        api = get_api()
        
        # Try different methods to get commit history
        # Method 1: Try list_repo_commits if available
        try:
            if hasattr(api, 'list_repo_commits'):
                commits = api.list_repo_commits(repo_id=DATASET_REPO, repo_type="dataset")
                
                # Look for merge commits (they contain "Merge pending submissions" in the message)
                for commit in commits:
                    commit_msg = getattr(commit, 'commit_message', None) or str(getattr(commit, 'message', ''))
                    if commit_msg and "Merge pending submissions" in commit_msg:
                        # Try to get commit date
                        commit_date = getattr(commit, 'created_at', None) or getattr(commit, 'date', None)
                        if commit_date:
                            if isinstance(commit_date, datetime):
                                return commit_date
                            elif isinstance(commit_date, str):
                                # Try to parse string date
                                try:
                                    return datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                                except:
                                    pass
                        
                        # Fallback: Parse timestamp from commit message
                        # Format: "Merge pending submissions (20240101T120000Z) — merged X entries"
                        match = re.search(r'\((\d{8}T\d{6}Z)\)', commit_msg)
                        if match:
                            try:
                                ts_str = match.group(1)
                                dt = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ")
                                return dt.replace(tzinfo=timezone.utc)
                            except ValueError:
                                pass
        except (AttributeError, TypeError) as e:
            # Method not available or incompatible format
            pass
        
        # Method 2: Try to get repo info and check for revision history
        # This is a fallback if commit listing doesn't work
        try:
            repo_info = api.repo_info(repo_id=DATASET_REPO, repo_type="dataset")
            # If repo has git integration, we might be able to get commits
            # For now, return None as we can't reliably get commit history this way
            pass
        except:
            pass
        
        return None
    except Exception:
        # Silently return None on any error - we'll handle it gracefully in the UI
        return None

def get_merge_status():
    """Get current merge status based on pending files and recent activity."""
    try:
        pending_count = get_pending_stories_count()
        if pending_count is None:
            return "Unknown", "Unable to determine status"
        
        if pending_count == 0:
            return "Idle", "No pending stories. All stories have been merged."
        
        # Check if there was a recent merge (within last hour)
        last_merge = get_last_merge_timestamp()
        if last_merge:
            now = datetime.now(timezone.utc)
            if isinstance(last_merge, datetime):
                time_diff = (now - last_merge.replace(tzinfo=timezone.utc) if last_merge.tzinfo is None else last_merge).total_seconds()
                if time_diff < 3600:  # Within last hour
                    return "Processing", f"Merge completed recently. {pending_count} new stories pending."
        
        return "Pending", f"{pending_count} stories waiting to be merged."
    except Exception as e:
        return "Error", f"Error determining status: {str(e)}"

def upload_jsonl_to_pending(story: str):
    try:
        api = get_api()
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

# Status Section
st.header("📊 Dataset Status")

# Refresh button at the top
refresh_clicked = st.button("🔄 Refresh Status", key="refresh_status")

# Initialize or refresh status data
if refresh_clicked or 'pending_count' not in st.session_state:
    with st.spinner("Fetching status information..."):
        st.session_state.pending_count = get_pending_stories_count()
        st.session_state.last_merge = get_last_merge_timestamp()
        st.session_state.merge_status, st.session_state.merge_message = get_merge_status()

# Use columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Pending Stories")
    pending_count = st.session_state.get('pending_count')
    if pending_count is not None:
        st.metric("Count", pending_count)
    else:
        st.error("Unable to fetch pending count")
        st.caption("Check your Hugging Face token and repository access")

with col2:
    st.subheader("Last Merge")
    last_merge = st.session_state.get('last_merge')
    if last_merge:
        if isinstance(last_merge, datetime):
            # Normalize timezone
            if last_merge.tzinfo is None:
                last_merge = last_merge.replace(tzinfo=timezone.utc)
            
            # Format for display
            display_time = last_merge.strftime("%Y-%m-%d %H:%M:%S UTC")
            st.metric("Timestamp", display_time)
            
            # Show relative time
            now = datetime.now(timezone.utc)
            time_diff = now - last_merge
            if time_diff.days > 0:
                st.caption(f"{time_diff.days} day(s) ago")
            elif time_diff.seconds >= 3600:
                hours = time_diff.seconds // 3600
                st.caption(f"{hours} hour(s) ago")
            else:
                minutes = time_diff.seconds // 60
                st.caption(f"{minutes} minute(s) ago")
        else:
            st.metric("Timestamp", str(last_merge))
    else:
        st.info("No merge found")
        st.caption("Merge history may not be available")

with col3:
    st.subheader("Merge Status")
    status = st.session_state.get('merge_status', 'Unknown')
    message = st.session_state.get('merge_message', '')
    
    if status == "Idle":
        st.success(f"✅ {status}")
    elif status == "Processing":
        st.info(f"🔄 {status}")
    elif status == "Pending":
        st.warning(f"⏳ {status}")
    else:
        st.error(f"❌ {status}")
    
    st.caption(message)

st.divider()

# Story Submission Section
st.header("✍️ Submit Your Story")
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
            # Refresh status after submission
            st.session_state.pending_count = get_pending_stories_count()
            st.session_state.merge_status, st.session_state.merge_message = get_merge_status()
            st.rerun()
        except Exception as e:
            st.error(f"Upload failed: {e}")
