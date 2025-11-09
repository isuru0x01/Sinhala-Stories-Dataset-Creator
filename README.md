# Sinhala Stories Dataset Creator - Streamlit App 📚

A web application built with Streamlit for collecting and managing Sinhala stories in a Hugging Face dataset.

## Features 🌟

- Submit Sinhala stories through a user-friendly interface
- Automatic validation of story content and length
- Duplicate story detection
- Real-time character counting and preview
- Dataset statistics visualization
- Direct integration with Hugging Face Datasets

## Requirements 📋

- Python 3.7+
- Streamlit
- Hugging Face `datasets` and `huggingface_hub` libraries
- Pandas

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

3. Configure Hugging Face token:
   - Create a `.streamlit/secrets.toml` file
   - Add your Hugging Face token:
     ```toml
     HUGGINGFACE_TOKEN = "your_token_here"
     ```

## Usage 💻

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided URL (typically `http://localhost:8501`)

3. Start submitting stories:
   - Enter your Sinhala story in the text area
   - Preview your story
   - Click "Submit Story" to add it to the dataset

## Dataset Structure 📊

Stories are stored in the [Isuru0x01/sinhala_stories](https://huggingface.co/datasets/Isuru0x01/sinhala_stories) dataset on Hugging Face with the following features:
- `story`: The main story text in Sinhala

## Validation Rules ✅

- Minimum story length: 50 characters
- Maximum story length: 50,000 characters
- Must contain Sinhala characters
- Duplicate detection based on content similarity

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