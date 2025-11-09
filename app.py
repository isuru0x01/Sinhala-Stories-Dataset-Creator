import os
import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, CommitOperationAdd
from datasets import load_dataset, Dataset
import time
from datetime import datetime
import hashlib
import json

# ===== CONFIGURATION =====
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]  # Remove fallback value to ensure we only use the secret
DATASET_REPO = 'Isuru0x01/sinhala_stories'

# Debug: Print first few characters of token
st.sidebar.write("Token preview:", f"{HUGGINGFACE_TOKEN[:8]}...")

MIN_STORY_LENGTH = 50  # Minimum characters
MAX_STORY_LENGTH = 50000  # Maximum characters

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Sinhala Story Submission",
    page_icon="📚",
    layout="wide"
)

# ===== HELPER FUNCTIONS =====
def validate_story(story):
    """Validate the story input"""
    errors = []
    
    if not story or not story.strip():
        errors.append("❌ Story cannot be empty")
    elif len(story.strip()) < MIN_STORY_LENGTH:
        errors.append(f"❌ Story must be at least {MIN_STORY_LENGTH} characters (current: {len(story.strip())})")
    elif len(story.strip()) > MAX_STORY_LENGTH:
        errors.append(f"❌ Story exceeds maximum length of {MAX_STORY_LENGTH} characters")
    
    # Check if story contains Sinhala characters
    sinhala_pattern = r'[\u0D80-\u0DFF]'
    if not any(ord(char) >= 0x0D80 and ord(char) <= 0x0DFF for char in story):
        errors.append("⚠️ Warning: No Sinhala characters detected")
    
    return errors

def check_duplicate(story, api, repo_id, threshold=0.9):
    """Check if story is similar to existing ones using efficient API calls"""
    try:
        # Only fetch the last part of the dataset
        dataset = load_dataset(repo_id, split="train[-100:]", token=HUGGINGFACE_TOKEN)
        
        # Check exact duplicates by comparing first 200 characters
        story_preview = story.strip()[:200]
        for existing_story in dataset['story']:
            if story_preview in existing_story[:200]:
                return True
        return False
    except Exception:
        # If there's an error loading the dataset slice, skip duplicate check
        return False

def get_statistics(dataset):
    """Get dataset statistics"""
    total_stories = len(dataset)
    total_chars = sum(len(story) for story in dataset['story'])
    avg_length = total_chars / total_stories if total_stories > 0 else 0
    
    return {
        'total_stories': total_stories,
        'total_chars': total_chars,
        'avg_length': int(avg_length)
    }

# ===== UI COMPONENTS =====
st.title('📚 Sinhala Story Submission')
st.markdown("---")

# Sidebar with instructions and statistics
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown(f"""
    1. Write your story in Sinhala
    2. Minimum length: **{MIN_STORY_LENGTH}** characters
    3. Maximum length: **{MAX_STORY_LENGTH}** characters
    4. Click Submit to add to the dataset
    """)
    
    st.markdown("---")
    st.header("📊 Dataset Statistics")
    
    if st.button("Refresh Stats"):
        with st.spinner("Loading statistics..."):
            try:
                dataset = load_dataset(DATASET_REPO, split="train")
                stats = get_statistics(dataset)
                st.metric("Total Stories", f"{stats['total_stories']:,}")
                st.metric("Total Characters", f"{stats['total_chars']:,}")
                st.metric("Average Length", f"{stats['avg_length']:,}")
            except Exception as e:
                st.error(f"Failed to load stats: {str(e)}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Initialize session state for story
    if 'story_text' not in st.session_state:
        st.session_state.story_text = ""
    
    story = st.text_area(
        'Enter your story here (Sinhala)',
        value=st.session_state.story_text,
        height=400,
        placeholder="ඔබේ කතාව මෙහි ලියන්න...",
        help=f"Write your story in Sinhala. Length: {MIN_STORY_LENGTH}-{MAX_STORY_LENGTH} characters"
    )
    
    # Character count
    char_count = len(story.strip())
    if char_count > 0:
        color = "green" if MIN_STORY_LENGTH <= char_count <= MAX_STORY_LENGTH else "red"
        st.markdown(f"Character count: <span style='color:{color}'>{char_count:,}</span> / {MAX_STORY_LENGTH:,}", unsafe_allow_html=True)

with col2:
    st.subheader("Preview")
    if story.strip():
        st.info(story[:300] + ("..." if len(story) > 300 else ""))
    else:
        st.write("Your story preview will appear here...")

# Submission buttons
col_submit, col_clear = st.columns([1, 1])

with col_submit:
    submit_button = st.button('✅ Submit Story', type="primary", use_container_width=True)

with col_clear:
    if st.button('🗑️ Clear', use_container_width=True):
        st.session_state.story_text = ""
        st.rerun()

# Handle submission
if submit_button:
    # Validate story
    validation_errors = validate_story(story)
    
    if validation_errors:
        for error in validation_errors:
            st.error(error)
    else:
        # Check token
        if not HUGGINGFACE_TOKEN:
            st.error("❌ Hugging Face token not found in secrets!")
            st.stop()
        
        try:
            with st.spinner('Submitting your story...'):
                progress = st.progress(0)
                status_text = st.empty()
                
                # Initialize Hugging Face API
                api = HfApi()
                
                # Step 1: Check for duplicates
                status_text.text("🔍 Checking for duplicates...")
                if check_duplicate(story, api, DATASET_REPO):
                    st.warning("⚠️ This story appears to be very similar to an existing one!")
                    if not st.checkbox("Submit anyway?"):
                        st.stop()
                progress.progress(0.3)
                
                # Step 2: Prepare new entry
                status_text.text("📝 Creating new entry...")
                new_story = {"story": story.strip()}
                
                # Create a temporary jsonl file for the new entry
                temp_file = "new_entry.jsonl"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(new_story, f, ensure_ascii=False)
                    f.write("\n")
                
                progress.progress(0.6)
                
                # Step 3: Upload directly to the hub
                status_text.text("☁️ Uploading to Hugging Face...")
                
                try:
                    # First try to download existing file
                    existing_content = api.download_file(
                        repo_id=DATASET_REPO,
                        path_in_repo="data/new_entries.jsonl",
                        token=HUGGINGFACE_TOKEN
                    )
                    # Append new content to existing content
                    with open(temp_file, 'rb') as f:
                        new_content = f.read()
                    combined_content = existing_content + new_content
                    
                    # Write combined content to temp file
                    with open(temp_file, 'wb') as f:
                        f.write(combined_content)
                except Exception:
                    # If file doesn't exist, just use the new content
                    pass
                
                # Upload the file
                operations = [
                    CommitOperationAdd(
                        path_in_repo="data/new_entries.jsonl",
                        path_or_fileobj=temp_file
                    )
                ]
                
                api.create_commit(
                    repo_id=DATASET_REPO,
                    operations=operations,
                    commit_message=f"Add new story via Streamlit app ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
                    token=HUGGINGFACE_TOKEN
                )
                
                # Clean up temporary file
                os.remove(temp_file)
                progress.progress(1.0)
                
                # Success message
                status_text.empty()
                progress.empty()
                st.success("✅ Story successfully submitted!")
                st.balloons()
                
                # Show submission details
                st.info(f"""
                **Submission Details:**
                - Characters: {len(story.strip()):,}
                - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """)
                
                # Clear the text area
                st.session_state.story_text = ""
                time.sleep(2)
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error submitting story: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Dataset: <a href='https://huggingface.co/datasets/Isuru0x01/sinhala_stories' target='_blank'>Isuru0x01/sinhala_stories</a></p>
    <p>Built with Streamlit • Powered by Hugging Face 🤗</p>
</div>
""", unsafe_allow_html=True)