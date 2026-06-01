Explainable Fake News Detector

Live Demo: https://explainable-fake-news-detector.streamlit.app/

An Explainable AI-powered Fake News Detection system that identifies whether a news article is REAL or FAKE using Natural Language Processing (NLP), TF-IDF vectorization, linguistic feature engineering, and a Random Forest classifier.

Unlike traditional black-box models, this project integrates LIME (Local Interpretable Model-Agnostic Explanations) to provide transparent and interpretable predictions, helping users understand why a particular article was classified as misinformation.

Features:
1. Real vs Fake news classification
2. TF-IDF text vectorization
3. Linguistic feature engineering
4. Random Forest classifier
5. Explainable AI using LIME
6. Confidence score prediction
7. Interactive Streamlit interface
8. Real-time prediction analysis
9. User-friendly web deployment

Dataset

Dataset used:

Fake and Real News Dataset

Source:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

ML Pipeline:
1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. Feature Engineering
5. TF-IDF Vectorization
6. Linguistic Feature Extraction
7. Feature Combination
8. Random Forest Training
9. Model Evaluation
10. Explainability using LIME
11. Streamlit Deployment

Linguistic Features Used:

The model combines TF-IDF features with handcrafted linguistic features:

1. Readability Score
2. Exclamation Count
3. Capital Letter Ratio
4. Average Word Length

These features improve the model’s ability to identify sensationalized and misleading content.

Explainable AI:

This project uses LIME (Local Interpretable Model-Agnostic Explanations) to explain individual predictions.

LIME highlights:

1. Important words influencing predictions
2. Positive and negative contributions
3. Why an article is classified as fake or real

This makes the system more transparent and trustworthy.

Future Improvements: 
1. SHAP Explainability
2. Deep Learning Models (LSTM/BERT)
3. News URL Analysis
4. Fact-Checking API Integration
5. Browser Extension
6. Multilingual Fake News Detection
7. News Source Credibility Scoring

Project Highlights: 

1. Explainable AI System
2. Trustworthy Machine Learning
3. NLP-Based Misinformation Detection
4. Real-World Deployment
5. Interpretable Predictions
6. End-to-End AI Application
