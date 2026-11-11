# Sinhala Stories Dataset Creator 📚

[![Automated Tests](https://github.com/isuru0x01/Sinhala-Stories-Dataset-Creator/actions/workflows/tests.yml/badge.svg)](https://github.com/isuru0x01/Sinhala-Stories-Dataset-Creator/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-sinhala__stories-blue)](https://huggingface.co/datasets/Isuru0x01/sinhala_stories)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21195380.svg)](https://doi.org/10.5281/zenodo.21195380)

A fault-tolerant, conflict-free crowdsourcing and incremental Apache Parquet ingestion framework for low-resource natural language processing (NLP) corpora on the Hugging Face Hub. 

Although its flagship production deployment curates the **10.9-million-row** [`Isuru0x01/sinhala_stories`](https://huggingface.co/datasets/Isuru0x01/sinhala_stories) corpus, the software is fully modular and generalizable: researchers and language communities can configure it for any low-resource script (e.g., Tamil, Dhivehi, Khmer, Swahili) via `config.yaml`.

---

## Key Features 🌟

- **Conflict-Free Asynchronous Staging:** Submissions are staged as isolated micro-files in a remote `pending/` directory, preventing Git tree write lock collisions during concurrent crowdsourcing.
- **Incremental Apache Parquet Ingestion:** A scheduled merge engine (`merge_pending_into_main.py`) batches pending files, aligns schemas, and appends records directly to an auxiliary delta partition (`data/train-append.parquet`), eliminating the need to load the 10.9M+ row core corpus into memory.
- **Decoupled Quality Validation & Heuristics:** Clean, headless Python module (`validator.py`) implementing Unicode NFC canonicalization, script ratio bounding, `langid` confidence classification, and heuristic spam scoring.
- **Two-Tier Deduplication:** $O(1)$ exact cryptographic SHA-256 fingerprint matching across the full corpus combined with word-level Jaccard lexical similarity against unmerged queue submissions.
- **Fault-Tolerant Offline Queue:** Automatic local disk queue fallback with seamless background replay when network connectivity is lost.
- **Declarative Generalizability:** Script boundaries, length bounds, language identifiers, and repository targets are externalized in `config.yaml`.
- **Automated Test Suite:** Comprehensive `pytest` test suite executed via GitHub Actions CI across Python 3.10, 3.11, and 3.12.

---

## System Architecture 🏗️

The ingestion system operates as a decoupled three-tier pipeline:

```
[ Contributor Web Client (app.py) ]
               │
               ▼  (Validation via validator.py)
[ Staged Queue: Hugging Face pending/ micro-jsonl files ]
               │
               ▼  (Every 5 min via merge_pending_into_main.py)
[ Incremental Parquet Partition (data/train-append.parquet) + hashes.txt + stats ]
```

1. **Client Submission Interface (`app.py`):** Collects contributor stories, runs pre-commit quality checks, computes real-time telemetry (reading time, word/character metrics), and writes isolated micro-files.
2. **Conflict-Free Staging Queue (Hugging Face Hub):** Submissions are committed as atomic micro-files in `pending/`, decoupling contributor write latency and eliminating concurrent write collisions.
3. **Automated Merge Engine (`merge_pending_into_main.py`):** Scheduled GitHub Actions runner that batch-fetches pending files, normalizes schemas, appends records to `data/train-append.parquet`, updates `hashes.txt` and `dataset_stats.json`, and atomically cleans up merged pending files.

---

## Repository Structure 📂

```
Sinhala-Stories-Dataset-Creator/
├── .github/
│   └── workflows/
│       ├── merge_hf_dataset.yml   # Scheduled Hugging Face merge cron job
│       └── tests.yml              # Automated pytest CI workflow across Python 3.10-3.12
├── paper/                         # JOSS manuscript directory
│   ├── paper.md                   # Complete manuscript (1,000–1,500 words)
│   ├── paper.bib                  # BibTeX references
│   ├── figure1_architecture.png   # Figure 1: Pipeline architecture diagram
│   └── figure2_distribution.png   # Figure 2: Corpus length distribution histogram
├── tests/                         # Automated unit test suite
│   ├── __init__.py
│   └── test_validation.py         # Tests for NFC, script ratio, spam, Jaccard, & hashes
├── app.py                         # Streamlit client submission interface
├── validator.py                   # Decoupled pure Python validation & quality heuristics
├── merge_pending_into_main.py     # Incremental Parquet append & queue purge engine
├── config.yaml                    # Declarative configuration parameters
├── pyproject.toml                 # Standard Python packaging metadata
├── requirements.txt               # Pinned project dependencies
├── LICENSE                        # MIT License
├── CONTRIBUTING.md                # Community guidelines for issues, support, & PRs
└── README.md                      # Documentation & usage guide
```

---

## Configuration (`config.yaml`) ⚙️

All project parameters, validation gates, and storage paths are fully customizable:

```yaml
project:
  name: "Sinhala Stories Dataset Creator"
  app_version: "2.0"
  validation_version: "1.2"
  hf_repo_id: "Isuru0x01/sinhala_stories"
  id_prefix: "SLS"

validation:
  min_length: 50
  max_length: 50000
  target_language_iso: "si"
  min_language_confidence: 0.90
  unicode_script_name: "Sinhala"
  unicode_range_start: 0x0D80  # U+0D80
  unicode_range_end: 0x0DFF    # U+0DFF
  min_script_ratio: 0.90
  min_spam_score: 0.20
  jaccard_similarity_threshold: 0.85

storage:
  pending_dir: "pending"
  local_queue_dir: "local_queue"
  append_parquet_path: "data/train-append.parquet"
  hashes_path: "hashes.txt"
  stats_path: "dataset_stats.json"
```

### Adapting to Other Low-Resource Languages
To adapt this software to another language (e.g., Tamil):
1. Change `target_language_iso` to `"ta"`.
2. Change `unicode_script_name` to `"Tamil"`.
3. Set `unicode_range_start: 0x0B80` and `unicode_range_end: 0x0BFF`.
4. Update `hf_repo_id` with your target Hugging Face repository.

---

## Installation & Setup 🛠️

### Prerequisites
- Python 3.10 or higher
- A Hugging Face account and personal access token with write permissions

### 1. Clone the repository
```bash
git clone https://github.com/isuru0x01/Sinhala-Stories-Dataset-Creator.git
cd Sinhala-Stories-Dataset-Creator
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
Install standard dependencies:
```bash
pip install -r requirements.txt
```
Or install in editable mode with development dependencies:
```bash
pip install -e ".[dev]"
```

---

## Running Automated Tests 🧪

To run the unit test suite locally:
```bash
pytest -v
```

All validation rules, Unicode normalization checks, script ratio calculations, spam filters, near-duplicate Jaccard logic, and SHA-256 digests are tested automatically without requiring a network connection or a running Streamlit server.

---

## Usage 💻

### Running the Web Interface
1. Set up your Hugging Face API token in `.streamlit/secrets.toml`:
   ```toml
   HUGGINGFACE_TOKEN = "hf_your_personal_access_token_here"
   ```
2. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open `http://localhost:8501` in your browser to submit stories.

### Running the Incremental Merge Script Manually
```bash
export HF_TOKEN="hf_your_personal_access_token_here"
python merge_pending_into_main.py
```

---

## Dataset Schema 📊

Records in the curated corpus contain the following features:

| Field | Type | Description |
| :--- | :--- | :--- |
| `story` | string | Normalized story text |
| `timestamp_utc` | string | Submission timestamp (UTC ISO format) |
| `story_length` | int | Total character count of normalized text |
| `sha256` | string | Cryptographic SHA-256 fingerprint |
| `submission_id` | string | Unique submission identifier (`PREFIX-YYYYMMDD-HEX`) |
| `language` | string | Detected ISO language code |
| `language_probability` | float | Confidence score from language detector ($\ge 0.90$) |
| `sinhala_percentage` | float | Percentage of characters in target Unicode script |
| `unique_characters` | int | Count of distinct Unicode characters |
| `average_word_length` | float | Average character length per word |
| `sentence_count` | int | Number of sentences |
| `paragraph_count` | int | Number of paragraphs |
| `contributor_session_hash` | string | Anonymized SHA-256 hash of contributor session UUID |
| `consent_given` | bool | Contributor confirmation of originality & public release |
| `is_ai_suspected` | bool | Rule-based flag for suspected synthetic text |
| `adult` | bool | Contributor-tagged adult/mature theme flag |
| `violence` | bool | Contributor-tagged graphic violence flag |
| `hate` | bool | Contributor-tagged offensive language flag |

---

## Contributing 🤝

Contributions, issue reports, and community adaptations are welcome! Please consult [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on reporting bugs, requesting features, and submitting pull requests.

---

## License 📄

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Citation 📖

If you use this software or the Sinhala Stories dataset in your research, please cite:

```bibtex
@article{samarappulige2026sinhala,
  title={{Sinhala-Stories-Dataset-Creator: A Fault-Tolerant Crowdsourcing and Incremental Parquet Ingestion Framework for Low-Resource NLP}},
  author={Samarappulige, Isuru and Ekanayake, Kaushalya and Uyangoda, Hasandi},
  journal={Journal of Open Source Software},
  year={2026}
}
```