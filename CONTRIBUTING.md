# Contributing to Sinhala-Stories-Dataset-Creator

We welcome contributions from researchers, developers, and language communities interested in extending this crowdsourced data ingestion pipeline to Sinhala and other low-resource languages!

## 1. Reporting Issues or Seeking Support
* **Bug Reports & Feature Requests:** Please open an issue on the [GitHub Issues tracker](https://github.com/isuru0x01/Sinhala-Stories-Dataset-Creator/issues). Include your Python version, operating system, and detailed steps to reproduce the issue.
* **Support & Questions:** For questions regarding adapting `config.yaml` to another low-resource language script or configuring the Hugging Face Hub CI/CD workflow, open a GitHub Discussion or Issue labeled `question`.

## 2. How to Contribute Code
1. **Fork the repository** on GitHub and create a feature branch:
   ```bash
   git checkout -b feature/my-improvement
   ```
2. **Set up a local development environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```
3. **Run the automated test suite before submitting changes:**
   ```bash
   pytest -v
   ```
4. **Generalizing or Extending Language Support:**
   If you are configuring the framework for a new language:
   - Update `config.yaml` with your language's ISO code, Unicode script boundaries, and validation thresholds.
   - Add relevant unit tests in `tests/test_validation.py` to ensure script coverage and language-specific quality heuristics.
5. **Submit a Pull Request (PR):**
   - Provide a clear, detailed summary of the modifications.
   - Ensure all automated GitHub Actions CI checks (`.github/workflows/tests.yml`) pass.
