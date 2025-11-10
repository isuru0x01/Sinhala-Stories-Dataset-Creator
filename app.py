# streamlit_safe_append.py
import streamlit as st
from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub.utils import HfHubHTTPError
from datetime import datetime, timezone
import io, json, re
import traceback

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
    """
    Get the number of pending stories.
    Counts files in pending/ directory that are waiting to be merged into the dataset.
    Once merged, stories are immediately available in the dataset (no status tracking needed).
    """
    try:
        api = get_api()
        
        # Count files in pending/ directory (waiting to be merged)
        files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
        pending_files_count = len([f for f in files if f.startswith(PENDING_DIR + "/") and f.endswith(".jsonl")])
        
        return pending_files_count
        
    except Exception as e:
        # Don't show error here - return None and handle in UI
        return None

def get_last_merge_timestamp():
    """
    Get the timestamp of the last successful merge.
    Attempts to retrieve commit history from the Hugging Face repository.
    Returns (timestamp, error_message) tuple where timestamp is None if not found.
    """
    api = get_api()
    
    # Method 1: Try list_repo_commits
    try:
        if hasattr(api, 'list_repo_commits'):
            commits = api.list_repo_commits(repo_id=DATASET_REPO, repo_type="dataset")
            
            # Convert to list to iterate (it might be a generator or iterator)
            try:
                commits_list = list(commits) if commits else []
            except (TypeError, AttributeError):
                # If it's not iterable, try to access it directly
                commits_list = [commits] if commits else []
            
            # Look for merge commits (they contain "Merge pending submissions" in the message)
            for commit in commits_list:
                # Try multiple ways to get commit message
                commit_msg = None
                if hasattr(commit, 'commit_message'):
                    commit_msg = commit.commit_message
                elif hasattr(commit, 'message'):
                    commit_msg = commit.message
                elif isinstance(commit, dict):
                    commit_msg = commit.get('commit_message') or commit.get('message') or commit.get('title')
                else:
                    commit_msg = str(commit)
                
                if commit_msg and "Merge pending submissions" in commit_msg:
                    # Try multiple ways to get commit date
                    commit_date = None
                    
                    # Check various attribute names
                    for attr in ['created_at', 'date', 'timestamp', 'authored_date', 'committed_date']:
                        if hasattr(commit, attr):
                            commit_date = getattr(commit, attr)
                            break
                    
                    # If it's a dict, check dict keys
                    if commit_date is None and isinstance(commit, dict):
                        for key in ['created_at', 'date', 'timestamp', 'authored_date', 'committed_date']:
                            if key in commit:
                                commit_date = commit[key]
                                break
                    
                    # Process the date if found
                    if commit_date:
                        if isinstance(commit_date, datetime):
                            return commit_date, None
                        elif isinstance(commit_date, str):
                            try:
                                # Try ISO format
                                if 'T' in commit_date:
                                    dt = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                                    return dt, None
                                # Try other formats if needed
                            except (ValueError, AttributeError):
                                pass
                        elif isinstance(commit_date, (int, float)):
                            # Might be a Unix timestamp
                            try:
                                return datetime.fromtimestamp(commit_date, tz=timezone.utc), None
                            except (ValueError, OSError):
                                pass
                    
                    # Fallback: Parse timestamp from commit message
                    # Format: "Merge pending submissions (20240101T120000Z) — merged X entries"
                    match = re.search(r'\((\d{8}T\d{6}Z)\)', commit_msg)
                    if match:
                        try:
                            ts_str = match.group(1)
                            dt = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ")
                            return dt.replace(tzinfo=timezone.utc), None
                        except ValueError:
                            pass
            
            # No merge commits found
            return None, "No merge commits found in repository history"
            
    except HfHubHTTPError as e:
        # Handle specific HTTP errors
        if e.status_code == 401:
            return None, "Authentication failed. Check your Hugging Face token."
        elif e.status_code == 403:
            return None, "Access forbidden. Token may lack repository permissions."
        elif e.status_code == 404:
            return None, "Repository or commit endpoint not found."
        else:
            return None, f"HTTP error {e.status_code}: {str(e)}"
    except AttributeError as e:
        # Method doesn't exist or wrong structure
        return None, f"API method error: {str(e)}"
    except Exception as e:
        # Other errors - log for debugging
        error_msg = f"Error retrieving commits: {str(e)}"
        # Store in session state for debugging
        if 'last_merge_error' not in st.session_state:
            st.session_state.last_merge_error = error_msg
        return None, error_msg
    
    # If we get here, the method doesn't exist
    return None, "Commit listing method not available"

def get_merge_status():
    """Get current merge status based on pending files and recent activity."""
    try:
        pending_count = get_pending_stories_count()
        if pending_count is None:
            return "Unknown", "Unable to determine status"
        
        if pending_count == 0:
            return "Idle", "No files waiting to merge. All submitted stories are in the dataset."
        
        # Check if there was a recent merge (within last hour)
        last_merge, merge_error = get_last_merge_timestamp()
        if last_merge:
            now = datetime.now(timezone.utc)
            if isinstance(last_merge, datetime):
                time_diff = (now - last_merge.replace(tzinfo=timezone.utc) if last_merge.tzinfo is None else last_merge).total_seconds()
                if time_diff < 3600:  # Within last hour
                    return "Processing", f"Merge completed recently. {pending_count} new stories pending."
        
        return "Pending", f"{pending_count} files waiting to be merged into dataset."
    except Exception as e:
        return "Error", f"Error determining status: {str(e)}"

def upload_jsonl_to_pending(story: str):
    try:
        api = get_api()
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        payload = {
            "story": story.strip(),
            "timestamp_utc": timestamp
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
        st.session_state.last_merge, st.session_state.last_merge_error = get_last_merge_timestamp()
        st.session_state.merge_status, st.session_state.merge_message = get_merge_status()

# Use columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Pending Stories")
    pending_count = st.session_state.get('pending_count')
    if pending_count is not None:
        st.metric("Count", pending_count)
        if pending_count == 0:
            st.caption("No files waiting to merge")
        else:
            st.caption("Files in pending/ directory")
    else:
        st.error("Unable to fetch pending count")
        st.caption("Check your Hugging Face token and repository access")

with col2:
    st.subheader("Last Merge")
    last_merge = st.session_state.get('last_merge')
    last_merge_error = st.session_state.get('last_merge_error')
    
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
        # Show error message if available, otherwise generic message
        if last_merge_error:
            if "No merge commits found" in last_merge_error:
                st.info("No merges yet")
                st.caption("No merge commits found in repository")
            elif "Authentication" in last_merge_error or "401" in last_merge_error:
                st.error("Auth Error")
                st.caption(last_merge_error)
            elif "403" in last_merge_error or "forbidden" in last_merge_error.lower():
                st.warning("Access Denied")
                st.caption(last_merge_error)
            elif "404" in last_merge_error or "not found" in last_merge_error.lower():
                st.warning("Not Found")
                st.caption(last_merge_error)
            else:
                st.warning("Error")
                st.caption(last_merge_error)
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

# Debug information (expandable)
if st.session_state.get('last_merge_error'):
    with st.expander("🔍 Debug Information", expanded=False):
        st.write("**Last Merge Error:**")
        st.code(st.session_state.get('last_merge_error', 'No error recorded'))
        if st.button("Clear Error", key="clear_error"):
            st.session_state.last_merge_error = None
            st.rerun()

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
