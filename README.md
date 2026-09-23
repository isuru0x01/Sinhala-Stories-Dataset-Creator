# Sinhala Stories Dataset Creator - Streamlit App 📚

A web application built with Streamlit for collecting and managing Sinhala stories in a Hugging Face dataset.

## Features 🌟

- **Web Ingestion Interface:** Streamlit app for collecting crowdsourced Sinhala stories.
- **Multi-Tier Quality Validation:** Unicode NFC normalization, character length constraints, script ratio checks, and language identification (`langid`).
- **Two-Tier Deduplication:** Exact SHA-256 matching against the full dataset combined with Jaccard lexical similarity against the pending queue.
- **Conflict-Free Staging Architecture:** Decoupled ingestion into a remote `pending/` directory on Hugging Face Hub, eliminating concurrent Git write conflicts.
- **Fault-Tolerant Offline Queue:** Local disk queue fallback with automated replay upon reconnection.
- **Automated Merge Workflow:** Scheduled GitHub Actions runner that performs incremental Parquet appending, index synchronization, and atomic queue purges.
- **Real-Time Analytics & Telemetry:** Character/word counts, estimated reading time, Sinhala script percentage, and live corpus statistics.

## System Architecture 🏗️

The ingestion system operates as a three-component decoupled pipeline:
1. **Client Submission Interface (`app.py`):** Collects contributor input, applies pre-commit quality gates, enriches metadata, and packages submissions into isolated `.jsonl` files.
2. **Fault-Tolerant Staged File Queue (Hugging Face Hub):** Submissions are committed as atomic micro-files to `pending/`, decoupling write latency and avoiding concurrent write collisions across contributors.
3. **Automated Merge Workflow (`merge_pending_into_main.py`):** A scheduled GitHub Actions workflow (runs every 5 minutes) batch-fetches pending files, aligns schemas, appends records to `data/train-append.parquet` (bypassing the need to reload the 10.9M+ row core dataset), updates `hashes.txt` and `dataset_stats.json`, and atomically purges the merged pending files.

## Dataset Structure 📊

Stories are stored in the [Isuru0x01/sinhala_stories](https://huggingface.co/datasets/Isuru0x01/sinhala_stories) dataset on Hugging Face with the following features:

| Field | Type | Description |
| :--- | :--- | :--- |
| `story` | string | Normalized story text in Sinhala |
| `timestamp_utc` | string | Submission timestamp (UTC ISO format) |
| `story_length` | int | Total character count of normalized text |
| `sha256` | string | Cryptographic SHA-256 hash of normalized text |
| `submission_id` | string | Unique submission identifier (`SLS-YYYYMMDD-HEX`) |
| `language` | string | Detected ISO language code (must be `si`) |
| `language_probability` | float | Confidence score from language detector ($\ge 0.90$) |
| `sinhala_percentage` | float | Percentage of Sinhala script characters |
| `unique_characters` | int | Count of distinct Unicode characters |
| `average_word_length` | float | Average character length per word |
| `sentence_count` | int | Number of sentences |
| `paragraph_count` | int | Number of paragraphs |
| `contributor_session_hash`| string | Anonymized SHA-256 hash of session UUID |
| `consent_given` | bool | Contributor confirmation of originality & public release |
| `is_ai_suspected` | bool | Heuristic flag for suspected synthetic text |
| `adult` | bool | Optional contributor flag: adult/mature themes |
| `violence` | bool | Optional contributor flag: violence or graphic content |
| `hate` | bool | Optional contributor flag: hate speech or offensive terms |

## Validation & Deduplication Rules ✅

### 1. Quality & Length Bounds
- **Minimum Story Length:** 50 characters
- **Maximum Story Length:** 50,000 characters
- **Script Requirement:** Text must contain Sinhala characters with high script ratio
- **Language Identification:** Must be identified as Sinhala (`si`) with $\ge 90\%$ probability via `langid`
- **Text Normalization:** Automatically normalized to Unicode NFC with whitespace and excessive period cleanup

### 2. Two-Tier Deduplication Strategy
To avoid computationally prohibitive global pairwise comparisons across 10.9+ million records during real-time web ingestion, the platform implements a two-tier strategy:
- **Main Corpus (Exact Deduplication):** Compares the SHA-256 fingerprint of the incoming story against a pre-indexed `hashes.txt` file ($O(1)$ set lookup). Identical stories are rejected immediately.
- **Staged Queue (Near-Duplicate Screening):** Evaluates incoming submissions against unmerged stories in the `pending/` queue using word-level Jaccard similarity. Submissions with $> 85\%$ lexical overlap are flagged and rejected.

> **Note for Downstream Researchers:** While the ingestion pipeline eliminates exact duplicates and near-duplicates within the same submission window, cross-batch near-duplicates may remain in the 10.9M+ row corpus. Researchers training language models are encouraged to perform global offline near-duplicate deduplication (e.g., using MinHash LSH via `datasketch`).

## Requirements 📋

- Python 3.8+
- Streamlit
- Hugging Face `datasets` and `huggingface_hub`
- Pandas, NumPy, pyarrow
- `langid`

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/isuru0x01/Sinhala-Stories-Dataset-Creator-Streamlit-App.git
cd Sinhala-Stories-Dataset-Creator-Streamlit-App
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Hugging Face access token:
   - Create a `.streamlit/secrets.toml` file:
     ```toml
     HUGGINGFACE_TOKEN = "your_hf_token_here"
     ```

## Usage 💻

1. Run the Streamlit web application:
```bash
streamlit run app.py
```

2. Open your browser at `http://localhost:8501`.
3. Enter your story, review telemetry analytics, confirm originality consent, and submit.
4. (Optional) Run the merge script manually or via GitHub Actions:
```bash
export HF_TOKEN="your_hf_token_here"
python merge_pending_into_main.py
```

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Hugging Face](https://huggingface.co/)
- Sinhala story dataset contributors

## Contact 📬

Project Link: [https://github.com/isuru0x01/Sinhala-Stories-Dataset-Creator-Streamlit-App](https://github.com/isuru0x01/Sinhala-Stories-Dataset-Creator-Streamlit-App)