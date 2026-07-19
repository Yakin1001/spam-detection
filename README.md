# SMS Spam Detection using Deep Learning

A deep learning pipeline that classifies SMS messages as spam or ham (legitimate), built with TensorFlow/Keras.

## Overview
This project explores and models the classic SMS Spam Collection dataset, covering the full pipeline from exploratory data analysis to deep learning model training.

## What It Does
- Loads and cleans the SMS Spam Collection dataset (5,572 messages)
- Performs exploratory data analysis — label distribution, duplicate detection, word frequency
- Visualizes common spam vs. ham vocabulary using WordCloud
- Preprocesses text with tokenization and padded sequences
- Trains a deep learning model (Embedding + LSTM/Bidirectional LSTM layers) for binary classification
- Evaluates model performance on held-out test data

## Tech Stack
- Python, TensorFlow / Keras
- Pandas, NumPy for data handling
- Scikit-learn for train/test splitting
- Matplotlib, Seaborn, WordCloud for visualization

## Dataset
[SMS Spam Collection Dataset](https://raw.githubusercontent.com/Yakin1001/project/master/SMSSpamCollection) — 5,572 labeled SMS messages (ham/spam).

## Running It
This project was developed in Google Colab. To run locally:
\`\`\`bash
pip install tensorflow pandas numpy seaborn matplotlib wordcloud scikit-learn
python untitled10.py
\`\`\`

## Related Work
A more advanced version of this project — using Bidirectional LSTM with Hierarchical Attention Networks for long-text classification — was built as a follow-up, achieving 95% accuracy on longer email threads.
