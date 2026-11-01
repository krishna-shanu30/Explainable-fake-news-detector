# Explainable Fake News Detector

A beginner-friendly NLP project that classifies a pasted news article as **FAKE** or **REAL** using a saved TF-IDF vectorizer and Random Forest classifier. It uses **LIME** to show which words influenced an individual prediction.

> **Important limitation:** This is a machine-learning text classifier, not a fact-checking service. It does not verify claims against primary sources, check publication history, or establish whether a story is true. Its output reflects patterns in its training dataset and can be wrong. Always verify important claims with multiple reliable sources.

## What this project demonstrates

- Text cleaning with NLTK stopwords and WordNet lemmatization
- TF-IDF feature extraction
- Random Forest classification
- Class-probability display
- Local model explanations with LIME
- A Streamlit user interface
- Basic input validation and automated tests
- GitHub Actions test workflow

## Current model pipeline

The checked-in model was trained on TF-IDF features only. The production app therefore uses the same TF-IDF-only feature shape for both predictions and LIME explanations.

The original training script contains commented-out linguistic feature experiments, but those features were **not** included in the saved model's training matrix. They are not used by the application. This avoids a feature-dimension mismatch between training and inference.

The original model artifacts are:
- `fake_news_model.pkl`
- `tfidf_vector.pkl`

The model was trained using the Kaggle Fake and Real News Dataset:  
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Project structure

```text
.
├── app.py                         # Streamlit application
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Training-compatible text cleaning
│   ├── prediction.py               # Model loading and prediction
│   └── explainability.py            # LIME explanation
├── tests/
│   ├── test_preprocessing.py
│   └── test_prediction.py
├── .github/workflows/tests.yml     # CI tests
├── fake_news_model.pkl
├── tfidf_vector.pkl
├── requirements.txt
└── README.md
```

The file `Explainable Fake News Detector.py` is the original exploratory training script. It contains local dataset paths from the development machine; edit those paths before rerunning it. It is not needed to run the deployed app.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/krishna-shanu30/Explainable-fake-news-detector.git
cd Explainable-fake-news-detector
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies and NLTK data

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet
```

### 4. Start the application

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually http://localhost:8501.

## Run tests

```bash
pytest -q
```

The GitHub Actions workflow runs the tests on pushes and pull requests targeting `main`.

## Deploy with Streamlit Community Cloud

1. Push the repository to GitHub.
2. Sign in to https://share.streamlit.io/.
3. Create an app from this repository.
4. Select branch `main` and entrypoint `app.py`.
5. Deploy. The application installs dependencies from `requirements.txt`; the NLTK resources are downloaded during startup only if missing (the current app instead displays a clear setup error if unavailable).

Keep the model and vectorizer files in the repository root, or update the artifact paths in `src/prediction.py`.

## Responsible use and limitations

- A high model confidence is not proof that an article is true or false.
- The training dataset may contain source, topic, date, or writing-style biases.
- Performance on current events, regional reporting, non-English content, satire, and rewritten articles is unknown unless separately evaluated.
- Do not use this project for automated moderation, reputation decisions, or other high-impact decisions.
- Do not paste confidential or personally identifying information into a hosted demo.
- Only load trusted pickle/joblib model artifacts. Loading untrusted serialized files can execute malicious code.

## Next steps for a stronger portfolio project

1. Add a reproducible training pipeline that accepts a configurable dataset path instead of a local drive path.
2. Save and report a held-out evaluation: precision, recall, F1, confusion matrix, and class balance.
3. Add cross-validation and check for duplicate or near-duplicate articles across train and test splits.
4. Calibrate probabilities and evaluate calibration before presenting confidence as a reliable probability.
5. Evaluate on a newer, independent dataset to measure performance on unseen topics and publishers.
6. Add URL retrieval and trusted-source comparison only with careful source validation and transparent citations.
7. Track model/data versions and document limitations.

## Tech stack

Python · pandas · NumPy · NLTK · scikit-learn · TF-IDF · Random Forest · LIME · Streamlit · pytest · GitHub Actions
