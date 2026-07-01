# app.py
import streamlit as st
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from datetime import datetime, timezone
import io, json, re, os, time, secrets, uuid, hashlib, unicodedata
import traceback
import langid
from datasets import load_dataset
import numpy as np

# CONFIG
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
DATASET_REPO = "Isuru0x01/sinhala_stories"
PENDING_DIR = "pending"              # folder in repo where we append small jsonl files
MIN_STORY_LENGTH = 50
MAX_STORY_LENGTH = 50000
VALIDATION_VERSION = "1.2"
APP_VERSION = "2.0"
LOCAL_QUEUE_DIR = "local_queue"

st.set_page_config(page_title="Sinhala Story Submission (Safe Append)", layout="wide")
st.title("📚 Sinhala Story Submission — Safe Append")

# Core Utilities

def normalize_story(story: str) -> str:
    """Performs Unicode NFC normalization, collapses multiple dots, trims whitespace, and collapses multiple spaces."""
    normalized = unicodedata.normalize("NFC", story)
    # Collapse multiple consecutive dots into a single dot
    normalized = re.sub(r'\.{2,}', '.', normalized)
    normalized = normalized.strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def compute_sha256(text: str) -> str:
    """Computes the SHA-256 hash of the text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def calculate_sinhala_percentage(text: str) -> float:
    """Calculates the percentage of Sinhala characters over all alphabetic characters."""
    alphabetic_chars = [c for c in text if c.isalpha()]
    if not alphabetic_chars:
        return 0.0
    sinhala_chars = [c for c in alphabetic_chars if 0x0D80 <= ord(c) <= 0x0DFF]
    return len(sinhala_chars) / len(alphabetic_chars)

def detect_sinhala_language(text: str) -> tuple[str, float]:
    """Identifies the language using langid. Returns (lang_code, probability)."""
    try:
        ranks = langid.rank(text)
        if not ranks:
            return 'unknown', 0.0
        
        # Softmax on log-likelihoods of top 5 languages to get normalized probability
        top_ranks = ranks[:5]
        max_score = top_ranks[0][1]
        scores = np.array([r[1] for r in top_ranks])
        exp_scores = np.exp(scores - max_score)
        probs = exp_scores / np.sum(exp_scores)
        
        return top_ranks[0][0], float(probs[0])
    except Exception:
        return 'unknown', 0.0

def get_words(text: str) -> set:
    """Returns the set of words in the text (alphanumeric, lowercase)."""
    return set(re.findall(r'\w+', text.lower()))

def jaccard_similarity(set1: set, set2: set) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Spam detection heuristics
def compute_spam_score(text: str) -> tuple[float, list[str]]:
    """Combines several lightweight heuristics into a spam score and returns list of reasons."""
    reasons = []
    score = 0.0
    
    # 1. Repeated characters (e.g. any character repeated 5+ times consecutively)
    # Use finditer to score proportional to the length of the repetition
    for match in re.finditer(r'(.)\1{4,}', text):
        match_str = match.group(0)
        repetitions = len(match_str)
        score += 0.1 * repetitions
        reasons.append(f"Character '{match.group(1)}' repeated {repetitions} times")
        
    # 2. Repeated words (consecutive)
    words = text.split()
    if words:
        repeated_words_count = 0
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                repeated_words_count += 1
        if repeated_words_count > 0:
            score += 0.2 * min(5, repeated_words_count)
            reasons.append(f"Consecutive repeated words: {repeated_words_count} times")
            
    # 3. Repeated sentences
    sentences = [s.strip() for s in re.split(r'[.?!\u0964]', text) if s.strip()]
    if sentences:
        unique_sentences = set(sentences)
        dupes_count = len(sentences) - len(unique_sentences)
        if dupes_count > 0:
            ratio = dupes_count / len(sentences)
            if ratio > 0.2:
                score += ratio * 1.5
                reasons.append(f"High sentence repetition: {ratio:.1%}")
                
    # 4. URL count
    urls = re.findall(r'https?://[^\s]+|www\.[^\s]+', text)
    if urls:
        score += 0.5 * len(urls)
        reasons.append(f"URLs found: {len(urls)}")
        
    # 5. Emoji ratio
    emoji_count = len(re.findall(r'[\U00010000-\U0010ffff]', text))
    if len(text) > 0:
        emoji_ratio = emoji_count / len(text)
        if emoji_ratio > 0.1:
            score += emoji_ratio * 2.0
            reasons.append(f"High emoji ratio: {emoji_ratio:.1%}")
            
    # 6. Punctuation ratio
    punctuations = re.findall(r'[!,.:;??"\'()\-—\[\]{}「」\u0964]', text)
    if len(text) > 0:
        punc_ratio = len(punctuations) / len(text)
        if punc_ratio > 0.15:
            score += (punc_ratio - 0.15) * 3.0
            reasons.append(f"High punctuation ratio: {punc_ratio:.1%}")
            
    # 7. Whitespace ratio
    whitespaces = re.findall(r'\s', text)
    if len(text) > 0:
        space_ratio = len(whitespaces) / len(text)
        if space_ratio > 0.3 or space_ratio < 0.05:
            score += 0.3
            reasons.append(f"Abnormal whitespace ratio: {space_ratio:.1%}")
            
    # 8. Vocabulary diversity (Unique words / Total words)
    if len(words) >= 20:
        diversity = len(set(words)) / len(words)
        if diversity < 0.4:
            score += (0.4 - diversity) * 2.0
            reasons.append(f"Low vocabulary diversity: {diversity:.1%}")
            
    return score, reasons

# AI Generation heuristics
def check_ai_suspected(story: str) -> bool:
    """Flag suspicious machine-generated content using rule-based heuristics."""
    sentences = [s.strip() for s in re.split(r'[.?!\u0964]', story) if s.strip()]
    paragraphs = [p.strip() for p in story.split('\n') if p.strip()]
    
    if not sentences:
        return False
        
    unique_sentences = set(sentences)
    if len(sentences) > 3 and len(unique_sentences) / len(sentences) < 0.6:
        return True
        
    if len(paragraphs) >= 3:
        p_lengths = [len(p.split()) for p in paragraphs]
        if len(set(p_lengths)) == 1 and p_lengths[0] > 10:
            return True
            
    if len(sentences) >= 4:
        first_words = []
        for s in sentences:
            w = s.split()
            if w:
                first_words.append(w[0].lower())
        if first_words:
            most_common_word_count = max(first_words.count(w) for w in set(first_words))
            if most_common_word_count / len(sentences) > 0.5 and len(set(first_words)) > 1:
                return True

    return False

# Server-Side Logging & Client IP
def get_client_ip():
    """Extracts client IP address from streamlit request headers."""
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers()
        if headers:
            ip = headers.get("X-Forwarded-For", "").split(",")[0].strip()
            if not ip:
                ip = headers.get("X-Real-IP", "").strip()
            return ip if ip else "unknown"
    except Exception:
        pass
    return "unknown"

def log_error(error_msg: str, traceback_str: str, submission_size: int):
    """Logs error traceback and metadata to a local log file."""
    os.makedirs("logs", exist_ok=True)
    ip_addr = get_client_ip()
    ip_hash = hashlib.sha256(ip_addr.encode('utf-8')).hexdigest()[:8]
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ip_hash": ip_hash,
        "error": error_msg,
        "submission_size": submission_size,
        "traceback": traceback_str
    }
    try:
        with open("logs/error.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def log_performance(upload_time: float, api_latency: float):
    """Logs upload performance and API latency metrics."""
    os.makedirs("logs", exist_ok=True)
    perf_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "upload_time": upload_time,
        "api_latency": api_latency
    }
    try:
        with open("logs/perf.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(perf_entry) + "\n")
    except Exception:
        pass

# Dataset statistics computation
def compute_story_stats(text: str):
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    unique_chars = len(set(text))
    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
    
    sentences = [s.strip() for s in re.split(r'[.?!\u0964]', text) if s.strip()]
    sentence_count = len(sentences)
    
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    paragraph_count = len(paragraphs)
    
    sinhala_pct = calculate_sinhala_percentage(text)
    
    reading_time_mins = word_count / 140.0
    if reading_time_mins < 1.0:
        reading_time_str = f"{max(1, int(reading_time_mins * 60))} seconds"
    else:
        reading_time_str = f"{reading_time_mins:.1f} minutes"
        
    return {
        "char_count": char_count,
        "word_count": word_count,
        "unique_chars": unique_chars,
        "avg_word_len": avg_word_len,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "sinhala_pct": sinhala_pct,
        "reading_time_str": reading_time_str
    }

# API and Dataset caching
def get_api():
    """Get initialized HfApi instance."""
    return HfApi(token=HUGGINGFACE_TOKEN)

@st.cache_data(ttl=60)
def load_merge_stats():
    """Retrieves merge metadata from merge_stats.json in the repository."""
    try:
        local_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename="merge_stats.json",
            repo_type="dataset",
            token=HUGGINGFACE_TOKEN
        )
        with open(local_path, "r", encoding="utf-8") as f:
            return json.loads(f.read().strip())
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_main_dataset_hashes_and_stories():
    """Loads the main dataset and extracts exact hashes and word sets for duplicate checking."""
    hashes = set()
    stories_data = []
    
    start_time = time.time()
    try:
        ds = load_dataset(DATASET_REPO, split="train", token=HUGGINGFACE_TOKEN)
        api_latency = time.time() - start_time
        log_performance(0.0, api_latency)
        
        has_sha = "sha256" in ds.column_names
        
        for row in ds:
            story = row.get("story", "")
            if not story:
                continue
            h = row.get("sha256") if has_sha else None
            if not h:
                h = compute_sha256(normalize_story(story))
            hashes.add(h)
            stories_data.append((get_words(story), h))
    except Exception as e:
        log_error(f"Failed to load main dataset: {str(e)}", traceback.format_exc(), 0)
        
    return hashes, stories_data

@st.cache_data(ttl=30)
def list_pending_filenames():
    """Lists files in the pending/ folder on Hugging Face."""
    try:
        api = get_api()
        files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
        return [f for f in files if f.startswith(PENDING_DIR + "/") and f.endswith(".jsonl")]
    except Exception as e:
        log_error(f"Failed to list pending files: {str(e)}", traceback.format_exc(), 0)
        return []

def check_duplicate(normalized_story: str) -> tuple[bool, str]:
    """Checks if the story is a duplicate of a merged or pending story."""
    new_hash = compute_sha256(normalized_story)
    new_hash_prefix = new_hash[:8]
    new_words = get_words(normalized_story)
    
    # 1. Check pending stories by listing filenames
    pending_files = list_pending_filenames()
    for pf in pending_files:
        base = os.path.basename(pf)
        parts = base.replace(".jsonl", "").split("_")
        if len(parts) >= 3 and parts[2].lower() == new_hash_prefix.lower():
            return True, "❌ Story is identical to a pending story awaiting merge."
            
    # 2. Check main dataset
    main_hashes, main_stories = get_main_dataset_hashes_and_stories()
    if new_hash in main_hashes:
        return True, "❌ Story already exists in the dataset (Exact duplicate)."
        
    for existing_words, h in main_stories:
        sim = jaccard_similarity(new_words, existing_words)
        if sim > 0.85:
            return True, f"❌ Story is extremely similar to an existing story (Near-duplicate, similarity: {sim:.1%})."
            
    return False, ""

def check_local_queue_duplicate(normalized_story: str) -> tuple[bool, str]:
    """Checks if the story is already queued in the local folder."""
    new_hash = compute_sha256(normalized_story)
    if os.path.exists(LOCAL_QUEUE_DIR):
        for fn in os.listdir(LOCAL_QUEUE_DIR):
            if fn.endswith(".jsonl"):
                try:
                    with open(os.path.join(LOCAL_QUEUE_DIR, fn), "r", encoding="utf-8") as f:
                        data = json.loads(f.read().strip())
                        story_in_file = data.get("story", "")
                        h = data.get("sha256") or compute_sha256(normalize_story(story_in_file))
                        if h == new_hash:
                            return True, "❌ Story is already in local queue awaiting upload."
                except Exception:
                    pass
    return False, ""

# Local queue recovery

def queue_submission_locally(payload: dict, submission_id: str):
    """Saves the submission payload to local disk in case Hugging Face is down."""
    os.makedirs(LOCAL_QUEUE_DIR, exist_ok=True)
    filename = f"{LOCAL_QUEUE_DIR}/entry_{submission_id}.jsonl"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        log_error(f"Failed to write to local queue: {str(e)}", traceback.format_exc(), len(payload.get("story", "")))
        return False

def process_local_queue() -> int:
    """Attempts to upload all locally queued submissions to Hugging Face."""
    if not os.path.exists(LOCAL_QUEUE_DIR):
        return 0
    files = [f for f in os.listdir(LOCAL_QUEUE_DIR) if f.endswith(".jsonl")]
    if not files:
        return 0
        
    try:
        api = get_api()
    except Exception:
        return 0
        
    success_count = 0
    for fn in files:
        local_path = os.path.join(LOCAL_QUEUE_DIR, fn)
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                os.remove(local_path)
                continue
                
            payload = json.loads(content)
            timestamp = payload.get("timestamp_utc", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
            submission_id = payload.get("submission_id", "unknown")
            sha_prefix = payload.get("sha256", "00000000")[:8]
            
            suffix = submission_id.split('-')[-1]
            remote_filename = f"{PENDING_DIR}/entry_{timestamp}_{sha_prefix}_{suffix.lower()}.jsonl"
            
            buf = io.BytesIO(content.encode("utf-8") + b"\n")
            
            uploaded = False
            for attempt in range(3):
                try:
                    start_time = time.time()
                    api.create_commit(
                        repo_id=DATASET_REPO,
                        repo_type="dataset",
                        operations=[CommitOperationAdd(path_in_repo=remote_filename, path_or_fileobj=buf)],
                        commit_message=f"Add pending submission {timestamp} (from local queue)"
                    )
                    upload_time = time.time() - start_time
                    log_performance(upload_time, 0.0)
                    uploaded = True
                    break
                except Exception as e:
                    if attempt == 2:
                        raise e
                    time.sleep(2 ** attempt)
                    
            if uploaded:
                os.remove(local_path)
                success_count += 1
        except Exception as e:
            log_error(f"Failed to process queued file {fn}: {str(e)}", traceback.format_exc(), os.path.getsize(local_path) if os.path.exists(local_path) else 0)
            
    if success_count > 0:
        st.cache_data.clear()
        
    return success_count

# Validation & Submission Functions

def validate_story(story: str) -> list[str]:
    errors = []
    s = normalize_story(story)
    
    if not s:
        errors.append("❌ Story cannot be empty")
        return errors
        
    if len(s) < MIN_STORY_LENGTH:
        errors.append(f"❌ Must be at least {MIN_STORY_LENGTH} characters (Current: {len(s)})")
    elif len(s) > MAX_STORY_LENGTH:
        errors.append(f"❌ Exceeds maximum {MAX_STORY_LENGTH} characters (Current: {len(s)})")
        
    if not any(0x0D80 <= ord(c) <= 0x0DFF for c in s):
        errors.append("⚠️ No Sinhala characters detected")
        
    sinhala_pct = calculate_sinhala_percentage(s)
    if sinhala_pct < 0.9 and any(c.isalpha() for c in s):
        errors.append(f"⚠️ Sinhala character ratio over alphabetic characters is {sinhala_pct:.1%}, which is below the 90% threshold.")
        
    lang, prob = detect_sinhala_language(s)
    if lang != 'si' or prob < 0.9:
        errors.append(f"❌ Language identification failed. Top guess: {lang} with {prob:.1%} probability. Must be Sinhala ('si') with at least 90% confidence.")
        
    spam_score, spam_reasons = compute_spam_score(s)
    if spam_score > 1.0:
        errors.append(f"❌ Story rejected as potential spam (Spam Score: {spam_score:.2f}). Reasons: {', '.join(spam_reasons)}")
        
    is_local_dup, local_msg = check_local_queue_duplicate(s)
    if is_local_dup:
        errors.append(local_msg)
        
    is_dup, dup_msg = check_duplicate(s)
    if is_dup:
        errors.append(dup_msg)
        
    return errors

def get_pending_stories_count():
    """Counts files in pending/ directory waiting to be merged."""
    try:
        pending_files = list_pending_filenames()
        return len(pending_files)
    except Exception:
        return None

def get_last_merge_timestamp():
    """Attempts to retrieve commit history from the Hugging Face repository."""
    api = get_api()
    try:
        if hasattr(api, 'list_repo_commits'):
            commits = api.list_repo_commits(repo_id=DATASET_REPO, repo_type="dataset")
            try:
                commits_list = list(commits) if commits else []
            except (TypeError, AttributeError):
                commits_list = [commits] if commits else []
            
            for commit in commits_list:
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
                    commit_date = None
                    for attr in ['created_at', 'date', 'timestamp', 'authored_date', 'committed_date']:
                        if hasattr(commit, attr):
                            commit_date = getattr(commit, attr)
                            break
                    if commit_date is None and isinstance(commit, dict):
                        for key in ['created_at', 'date', 'timestamp', 'authored_date', 'committed_date']:
                            if key in commit:
                                commit_date = commit[key]
                                break
                    if commit_date:
                        if isinstance(commit_date, datetime):
                            return commit_date, None
                        elif isinstance(commit_date, str):
                            try:
                                if 'T' in commit_date:
                                    dt = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                                    return dt, None
                            except (ValueError, AttributeError):
                                pass
                        elif isinstance(commit_date, (int, float)):
                            try:
                                return datetime.fromtimestamp(commit_date, tz=timezone.utc), None
                            except (ValueError, OSError):
                                pass
                    
                    match = re.search(r'\((\d{8}T\d{6}Z)\)', commit_msg)
                    if match:
                        try:
                            ts_str = match.group(1)
                            dt = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ")
                            return dt.replace(tzinfo=timezone.utc), None
                        except ValueError:
                            pass
            return None, "No merge commits found in repository history"
    except HfHubHTTPError as e:
        if e.status_code == 401:
            return None, "Authentication failed. Check your Hugging Face token."
        elif e.status_code == 403:
            return None, "Access forbidden. Token may lack repository permissions."
        elif e.status_code == 404:
            return None, "Repository or commit endpoint not found."
        else:
            return None, f"HTTP error {e.status_code}: {str(e)}"
    except Exception as e:
        error_msg = f"Error retrieving commits: {str(e)}"
        if 'last_merge_error' not in st.session_state:
            st.session_state.last_merge_error = error_msg
        return None, error_msg
    return None, "Commit listing method not available"

def get_last_merge_timestamp_with_stats():
    """Retrieves last merge timestamp using merge_stats.json, falling back to commit history."""
    stats = load_merge_stats()
    if stats and "last_merge_timestamp" in stats:
        try:
            ts_str = stats["last_merge_timestamp"]
            dt = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ")
            return dt.replace(tzinfo=timezone.utc), None
        except Exception:
            pass
    return get_last_merge_timestamp()

def get_merge_status():
    """Get current merge status based on pending files and recent activity."""
    try:
        pending_count = get_pending_stories_count()
        if pending_count is None:
            return "Unknown", "Unable to determine status"
        
        if pending_count == 0:
            return "Idle", "No files waiting to merge. All submitted stories are in the dataset."
        
        last_merge, merge_error = get_last_merge_timestamp_with_stats()
        if last_merge:
            now = datetime.now(timezone.utc)
            if isinstance(last_merge, datetime):
                time_diff = (now - last_merge.replace(tzinfo=timezone.utc) if last_merge.tzinfo is None else last_merge).total_seconds()
                if time_diff < 3600:
                    return "Processing", f"Merge completed recently. {pending_count} new stories pending."
        
        return "Pending", f"{pending_count} files waiting to be merged into dataset."
    except Exception as e:
        return "Error", f"Error determining status: {str(e)}"

def upload_jsonl_to_pending(story: str, consent_given: bool, adult: bool, violence: bool, hate: bool):
    """Enriches the story metadata, generates a submission ID, and commits to Hugging Face or queues locally."""
    # 1. Normalize
    normalized = normalize_story(story)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    date_str = datetime.utcnow().strftime("%Y%m%d")
    
    # 2. Generate submission_id and session_id (for contributor session tracking)
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    contributor_session_hash = hashlib.sha256(st.session_state.session_id.encode('utf-8')).hexdigest()[:8]
    
    suffix = secrets.token_hex(3).upper() # 6 hex characters
    submission_id = f"SLS-{date_str}-{suffix}"
    
    # Calculate stats
    stats = compute_story_stats(normalized)
    sha256_hash = compute_sha256(normalized)
    
    # Detect language
    lang, prob = detect_sinhala_language(normalized)
    
    # Detect AI suspected
    is_ai = check_ai_suspected(normalized)
    
    payload = {
        "story": normalized,
        "timestamp_utc": timestamp,
        "story_length": len(normalized),
        "sha256": sha256_hash,
        "submission_id": submission_id,
        "app_version": APP_VERSION,
        "validation_version": VALIDATION_VERSION,
        "consent_given": consent_given,
        "is_ai_suspected": is_ai,
        "adult": adult,
        "violence": violence,
        "hate": hate,
        "language": lang,
        "language_probability": prob,
        "unique_characters": stats["unique_chars"],
        "average_word_length": stats["avg_word_len"],
        "sentence_count": stats["sentence_count"],
        "paragraph_count": stats["paragraph_count"],
        "sinhala_percentage": stats["sinhala_pct"],
        "contributor_session_hash": contributor_session_hash
    }
    
    # Generate filename
    sha_prefix = sha256_hash[:8]
    filename = f"{PENDING_DIR}/entry_{timestamp}_{sha_prefix}_{suffix.lower()}.jsonl"
    
    content_str = json.dumps(payload, ensure_ascii=False) + "\n"
    buf = io.BytesIO(content_str.encode("utf-8"))
    
    start_time = time.time()
    try:
        api = get_api()
        # Perform upload
        api.create_commit(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            operations=[CommitOperationAdd(path_in_repo=filename, path_or_fileobj=buf)],
            commit_message=f"Add pending submission {submission_id}"
        )
        upload_time = time.time() - start_time
        log_performance(upload_time, 0.0)
        return filename, submission_id, False # uploaded successfully, not queued
    except Exception as e:
        # Log the error
        tb_str = traceback.format_exc()
        log_error(f"HF upload failed for {submission_id}: {str(e)}", tb_str, len(normalized))
        
        # Save to local queue
        queued = queue_submission_locally(payload, submission_id)
        if queued:
            return filename, submission_id, True # saved to local queue
        else:
            raise e

@st.cache_data(ttl=300)
def calculate_dataset_stats():
    """Computes high-level analytics over the merged main dataset."""
    try:
        ds = load_dataset(DATASET_REPO, split="train", token=HUGGINGFACE_TOKEN)
        total_stories = len(ds)
        
        lengths = [len(row.get("story", "")) for row in ds]
        if lengths:
            total_size_chars = sum(lengths)
            avg_len = float(np.mean(lengths))
            median_len = float(np.median(lengths))
            longest_len = int(max(lengths))
        else:
            total_size_chars = 0
            avg_len = 0.0
            median_len = 0.0
            longest_len = 0
            
        now = datetime.now(timezone.utc)
        today_count = 0
        week_count = 0
        contributors = set()
        
        for row in ds:
            ts_str = row.get("timestamp_utc")
            if ts_str:
                try:
                    if 'T' in ts_str:
                        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    else:
                        dt = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                    
                    time_diff = now - dt
                    if time_diff.days == 0:
                        today_count += 1
                    if time_diff.days < 7:
                        week_count += 1
                except Exception:
                    pass
            
            session_hash = row.get("contributor_session_hash")
            if session_hash:
                contributors.add(session_hash)
            else:
                contributors.add(row.get("timestamp_utc", "")[:13]) # Fallback proxy grouping
                
        approx_contributors = len(contributors) if contributors else 0
        
        return {
            "total_stories": total_stories,
            "total_size_chars": total_size_chars,
            "avg_len": avg_len,
            "median_len": median_len,
            "longest_len": longest_len,
            "today_count": today_count,
            "week_count": week_count,
            "approx_contributors": approx_contributors
        }
    except Exception as e:
        log_error(f"Failed to calculate dataset stats: {str(e)}", traceback.format_exc(), 0)
        return None

def display_story_stats(text: str):
    """Renders character-level analytics and reading metrics for the submitted story."""
    stats = compute_story_stats(text)
    
    st.markdown("### 📊 Submitted Story Analytics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Characters", f"{stats['char_count']:,}")
    with col2:
        st.metric("Words", f"{stats['word_count']:,}")
    with col3:
        st.metric("Sinhala Ratio", f"{stats['sinhala_pct']:.1%}")
    with col4:
        st.metric("Est. Reading Time", stats["reading_time_str"])
        
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Unique Characters", f"{stats['unique_chars']:,}")
    with col6:
        st.metric("Avg Word Length", f"{stats['avg_word_len']:.1f}")
    with col7:
        st.metric("Sentences", f"{stats['sentence_count']:,}")
    with col8:
        st.metric("Paragraphs", f"{stats['paragraph_count']:,}")

def show_admin_page():
    """Renders the hidden admin page for health monitoring and log inspection."""
    st.header("⚙️ Admin Dashboard & Health Monitor")
    
    st.subheader("System Latency & Connectivity")
    start_time = time.time()
    try:
        api = get_api()
        api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
        latency = time.time() - start_time
        st.success(f"✅ Hugging Face API Connection: OK (Latency: {latency:.2f}s)")
    except Exception as e:
        latency = time.time() - start_time
        st.error(f"❌ Hugging Face API Connection: FAILED (Attempt duration: {latency:.2f}s). Error: {str(e)}")
        
    st.subheader("Performance Logs")
    perf_file = "logs/perf.log"
    if os.path.exists(perf_file):
        try:
            upload_times = []
            api_latencies = []
            with open(perf_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    ut = data.get("upload_time", 0.0)
                    al = data.get("api_latency", 0.0)
                    if ut > 0.0:
                        upload_times.append(ut)
                    if al > 0.0:
                        api_latencies.append(al)
            
            col1, col2 = st.columns(2)
            with col1:
                if upload_times:
                    st.metric("Avg Upload Time", f"{np.mean(upload_times):.2f}s")
                else:
                    st.metric("Avg Upload Time", "N/A")
            with col2:
                if api_latencies:
                    st.metric("Avg API Latency", f"{np.mean(api_latencies):.2f}s")
                else:
                    st.metric("Avg API Latency", "N/A")
        except Exception as e:
            st.error(f"Error parsing performance logs: {str(e)}")
    else:
        st.info("No performance log entries found yet.")
        
    st.subheader("Local Queue Status")
    queued_files = []
    if os.path.exists(LOCAL_QUEUE_DIR):
        queued_files = [f for f in os.listdir(LOCAL_QUEUE_DIR) if f.endswith(".jsonl")]
        
    if queued_files:
        st.warning(f"⏳ {len(queued_files)} submission(s) currently cached in local queue due to previous failures.")
        if st.button("Retry Uploading Queue Now"):
            with st.spinner("Processing local queue..."):
                success_count = process_local_queue()
                if success_count > 0:
                    st.success(f"Successfully uploaded {success_count} queued submission(s)!")
                    st.rerun()
                else:
                    st.error("Failed to upload queued items. Check connection.")
    else:
        st.success("✅ Local queue is empty. All submissions uploaded successfully.")
        
    st.subheader("Recent Error Logs")
    error_file = "logs/error.log"
    if os.path.exists(error_file):
        try:
            with open(error_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            recent_errors = [json.loads(l.strip()) for l in lines[-10:]]
            recent_errors.reverse()
            
            for idx, err in enumerate(recent_errors):
                with st.expander(f"⚠️ Error {idx+1}: {err.get('error', 'Unknown error')} ({err.get('timestamp', 'N/A')})"):
                    st.write(f"**IP Hash:** {err.get('ip_hash', 'N/A')}")
                    st.write(f"**Submission Size:** {err.get('submission_size', 0)} characters")
                    st.write("**Traceback:**")
                    st.code(err.get('traceback', 'No traceback available'))
            
            if st.button("Clear Error Log"):
                os.remove(error_file)
                st.success("Error log cleared!")
                st.rerun()
        except Exception as e:
            st.error(f"Error displaying error logs: {str(e)}")
    else:
        st.info("No error log entries found yet.")

# Main Application Execution Logic

# 1. Automatic recovery queue check on startup
if 'local_queue_processed' not in st.session_state:
    queued_uploaded = process_local_queue()
    if queued_uploaded > 0:
        st.info(f"🔄 Automatically uploaded {queued_uploaded} submission(s) from local cache!")
    st.session_state.local_queue_processed = True

# 2. Navigation Sidebar
st.sidebar.title("Navigation")
admin_mode = False
if st.query_params.get("admin") == "true" or st.query_params.get("page") == "admin":
    admin_mode = True
else:
    with st.sidebar.expander("Admin Portal"):
        admin_password = st.text_input("Enter Admin Password", type="password")
        correct_password = st.secrets.get("ADMIN_PASSWORD", "sinhala_admin")
        if admin_password == correct_password:
            admin_mode = True
            st.success("Admin mode unlocked!")

page = "Submit Story"
if admin_mode:
    page = st.sidebar.radio("Go to Page", ["Submit Story", "Admin Dashboard"])

if page == "Admin Dashboard":
    show_admin_page()
else:
    # ------------------ SUBMIT STORY PAGE ------------------
    st.header("📊 Dataset Status")
    
    refresh_clicked = st.button("🔄 Refresh Status", key="refresh_status")
    
    if refresh_clicked or 'pending_count' not in st.session_state or 'dataset_stats' not in st.session_state:
        with st.spinner("Fetching status information..."):
            st.session_state.pending_count = get_pending_stories_count()
            st.session_state.last_merge, st.session_state.last_merge_error = get_last_merge_timestamp_with_stats()
            st.session_state.merge_status, st.session_state.merge_message = get_merge_status()
            st.session_state.merge_stats = load_merge_stats()
            st.session_state.dataset_stats = calculate_dataset_stats()
            
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Dataset Metrics")
        ds_stats = st.session_state.get('dataset_stats')
        if ds_stats:
            st.metric("Total Stories (Merged)", f"{ds_stats['total_stories']:,}")
            st.metric("Total Size (Characters)", f"{ds_stats['total_size_chars']:,}")
            st.caption(f"Approximate Contributors: {ds_stats['approx_contributors']}")
        else:
            st.warning("Unable to fetch dataset metrics")
            st.caption("No data retrieved yet")
            
    with col2:
        st.subheader("Story Metrics")
        if ds_stats:
            st.metric("Average Story Length", f"{int(ds_stats['avg_len']):,} chars")
            st.metric("Median Story Length", f"{int(ds_stats['median_len']):,} chars")
            st.caption(f"Longest Story: {ds_stats['longest_len']:,} chars")
        else:
            st.warning("Story statistics unavailable")
            
    with col3:
        st.subheader("Submissions & Merge")
        pending_count = st.session_state.get('pending_count', 0)
        st.metric("Pending Stories", pending_count)
        
        status = st.session_state.get('merge_status', 'Unknown')
        message = st.session_state.get('merge_message', '')
        if status == "Idle":
            st.success(f"✅ Status: {status}")
        elif status == "Processing":
            st.info(f"🔄 Status: {status}")
        elif status == "Pending":
            st.warning(f"⏳ Status: {status}")
        else:
            st.error(f"❌ Status: {status}")
        st.caption(message)
        
    st.write("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Submission Activity")
        if ds_stats:
            st.write(f"📅 **Submitted Today:** {ds_stats['today_count']} stories")
            st.write(f"📅 **Submitted This Week:** {ds_stats['week_count']} stories")
        else:
            st.write("Activity data not available.")
            
    with col_b:
        st.subheader("Last Merge Activity")
        m_stats = st.session_state.get('merge_stats')
        last_merge = st.session_state.get('last_merge')
        
        if last_merge:
            st.write(f"⏰ **Last Merge Time:** {last_merge.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        else:
            st.write("⏰ **Last Merge Time:** No merges recorded")
            
        if m_stats:
            st.write(f"⏱️ **Last Merge Duration:** {m_stats.get('duration_seconds', 0.0):.2f} seconds")
            st.write(f"📦 **Merged Entries Count:** {m_stats.get('merged_entries_count', 0)} entries")
        else:
            st.write("⏱️ **Last Merge Duration:** N/A (No stats file found)")
            
    st.divider()
    
    # Submission Form Section
    st.header("✍️ Submit Your Story")
    
    with st.form("story_submission_form", clear_on_submit=False):
        story = st.text_area("Write your story (Sinhala)", height=300, placeholder="ඔබේ කතාව මෙහි ලියන්න...")
        
        st.markdown("### 📄 Contributor Consent")
        consent_1 = st.checkbox("I confirm this story is my own work or I have permission to contribute it.")
        consent_2 = st.checkbox("I understand this will become part of a public research dataset.")
        
        st.markdown("### ⚠️ Content Flags (Optional)")
        st.caption("Please check if your story contains any of the following to help downstream filtering:")
        adult = st.checkbox("Adult content / themes")
        violence = st.checkbox("Violence or graphic content")
        hate = st.checkbox("Hate speech or offensive language")
        
        submit_clicked = st.form_submit_button("Submit Story")
        
    if submit_clicked:
        if not (consent_1 and consent_2):
            st.error("❌ You must check both boxes in the Contributor Consent section before submitting.")
        else:
            errors = validate_story(story)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                norm_story = normalize_story(story)
                with st.spinner("Uploading story..."):
                    try:
                        filename, submission_id, is_queued = upload_jsonl_to_pending(
                            story,
                            consent_given=True,
                            adult=adult,
                            violence=violence,
                            hate=hate
                        )
                        
                        if is_queued:
                            st.warning("⚠️ Hugging Face API is currently unavailable. Your story has been saved in the local queue and will be uploaded automatically once the connection is restored.")
                            st.info(f"Submission ID: `{submission_id}`")
                        else:
                            st.success("✓ Upload completed")
                            st.success(f"✅ Submitted safely — stored in pending/ on Hugging Face. Submission ID: `{submission_id}`")
                            st.info(f"File created: `{filename}`")
                            
                        # Story Stats Presentation
                        display_story_stats(norm_story)
                        
                        # Refresh dashboard statistics
                        st.cache_data.clear()
                        st.session_state.pending_count = get_pending_stories_count()
                        st.session_state.merge_status, st.session_state.merge_message = get_merge_status()
                        st.session_state.dataset_stats = calculate_dataset_stats()
                        
                    except Exception as e:
                        st.error(f"Upload failed completely: {e}")
