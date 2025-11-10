import os
import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from datasets import load_dataset, Dataset
import time
from datetime import datetime
import hashlib
import json

# ===== CONFIGURATION =====
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]  # Remove fallback value to ensure we only use the secret
DATASET_REPO = 'Isuru0x01/sinhala_stories'

# Debug information in sidebar
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
    """Check if story is similar to existing ones using dataset streaming"""
    try:
        # Load only the last chunk of the dataset
        dataset = load_dataset(repo_id, split="train[-100:]", token=HUGGINGFACE_TOKEN)
        
        # Check for similar stories
        story_preview = story.strip()[:200]
        for existing_story in dataset['story']:
            if story_preview in existing_story[:200]:
                return True
        return False
    except Exception:
        # If there's an error checking duplicates, skip the check
        return False

def get_statistics(api, repo_id):
    """Get dataset statistics using dataset streaming"""
    try:
        # Load the dataset using streaming to minimize memory usage
        dataset = load_dataset(repo_id, streaming=True, token=HUGGINGFACE_TOKEN)
        
        total_stories = 0
        total_chars = 0
        
        # Process the dataset in chunks
        for batch in dataset['train'].iter(batch_size=100):
            if 'story' in batch:
                stories = batch['story']
                total_stories += len(stories)
                total_chars += sum(len(story) for story in stories)
        
        avg_length = total_chars / total_stories if total_stories > 0 else 0
        
        return {
            'total_stories': total_stories,
            'total_chars': total_chars,
            'avg_length': int(avg_length)
        }
    except Exception as e:
        st.error(f"Error calculating statistics: {str(e)}")
        return {
            'total_stories': 0,
            'total_chars': 0,
            'avg_length': 0
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
                # Load both main dataset and any new entries
                all_stories = []
                
                # Debug information container
                debug_info = []
                
                # Try to load the main dataset
                try:
                    dataset = load_dataset(DATASET_REPO, token=HUGGINGFACE_TOKEN)
                    debug_info.append("Successfully connected to dataset")
                    
                    # Iterate through all splits
                    for split_name in dataset.keys():
                        try:
                            if 'story' in dataset[split_name].features:
                                stories = dataset[split_name]['story']
                                all_stories.extend(stories)
                                debug_info.append(f"Found {len(stories)} stories in '{split_name}' split")
                        except Exception as split_e:
                            debug_info.append(f"Error loading split {split_name}: {str(split_e)}")
                            
                except Exception as e:
                    debug_info.append(f"Error loading main dataset: {str(e)}")
                
                # Try different file patterns
                api = HfApi()
                try:
                    # List all files in the repository
                    files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset", token=HUGGINGFACE_TOKEN)
                    debug_info.append(f"Files in repository: {files}")
                    
                    # Look for JSONL files
                    jsonl_files = [f for f in files if f.endswith('.jsonl')]
                    for jsonl_file in jsonl_files:
                        try:
                            # Download the file using hf_hub_download
                            local_file_path = hf_hub_download(
                                repo_id=DATASET_REPO,
                                filename=jsonl_file,
                                repo_type="dataset",
                                token=HUGGINGFACE_TOKEN
                            )
                            
                            # Read and parse the file
                            with open(local_file_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():  # Skip empty lines
                                        try:
                                            story_entry = json.loads(line)
                                            if 'story' in story_entry:
                                                all_stories.append(story_entry['story'])
                                        except json.JSONDecodeError:
                                            continue
                            debug_info.append(f"Processed JSONL file: {jsonl_file}")
                        except Exception as file_e:
                            debug_info.append(f"Error processing {jsonl_file}: {str(file_e)}")
                except Exception as e:
                    debug_info.append(f"Error listing repository files: {str(e)}")
                
                # Display debug information
                with st.expander("🔍 Debug Information"):
                    for info in debug_info:
                        st.text(info)                # Get statistics using the new streaming method
                api = HfApi(token=HUGGINGFACE_TOKEN)
                stats = get_statistics(api, DATASET_REPO)
                
                # Display stats
                st.metric("Total Stories", f"{stats['total_stories']:,}")
                st.metric("Total Characters", f"{stats['total_chars']:,}")
                st.metric("Average Length", f"{stats['avg_length']:,}")
                
                # Show data sources info
                st.info(f"📊 Statistics include stories from all sources in the dataset repository")
                
            except Exception as e:
                st.error(f"Failed to load stats: {str(e)}")
                st.exception(e)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
# Initialize session state for story input
    if "story_input" not in st.session_state:
        st.session_state.story_input = ""

    story = st.text_area(
        label='Enter your story here (Sinhala)',
        key="story_input",
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
        st.session_state.story_input = ""
        st.rerun()

# Handle submission
if submit_button:
    # Validate story
    validation_errors = validate_story(story)
    
    if validation_errors:
        for error in validation_errors:
            st.error(error)
    else:
        # Validate repository access
        try:
            api = HfApi(token=HUGGINGFACE_TOKEN)
            # Check repository access
            try:
                repo_info = api.repo_info(repo_id=DATASET_REPO, repo_type="dataset", token=HUGGINGFACE_TOKEN)
                st.success(f"✅ Connected to repository: {DATASET_REPO}")
            except Exception as repo_error:
                st.error(f"❌ Cannot access repository {DATASET_REPO}. Error: {str(repo_error)}")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Error connecting to Hugging Face. Please ensure your repository is accessible.")
            st.error(f"Detailed error: {str(e)}")
            st.stop()
        
        try:
            with st.spinner('Submitting your story...'):
                progress = st.progress(0)
                status_text = st.empty()
                
                # Initialize Hugging Face API
                api = HfApi(token=HUGGINGFACE_TOKEN)
                
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
                    # Load only the last chunk of the dataset to minimize memory usage
                    dataset = load_dataset(DATASET_REPO, token=HUGGINGFACE_TOKEN)
                    
                    # Get the last 100 stories from the dataset
                    existing_stories = list(dataset['train'].select(range(max(0, len(dataset['train']) - 100)))['story'])
                    
                    # Add the new story
                    new_story_dict = {"story": [story.strip()]}
                    new_story_dataset = Dataset.from_dict(new_story_dict)
                    
                    # Push the new story to the hub
                    new_story_dataset.push_to_hub(
                        DATASET_REPO,
                        token=HUGGINGFACE_TOKEN,
                        commit_message=f"Add new story via Streamlit app ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
                        branch="new-stories"  # Use a separate branch for new stories
                    )
                    
                    debug_info = "Successfully appended new story to the dataset"
                    st.write(debug_info)
                    
                except Exception as e:
                    st.error(f"Error updating dataset: {str(e)}")
                    raise e
                
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
                
                # Clear the text area immediately and rerun
                st.session_state.story_input = ""
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
