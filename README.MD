# Sentiment Analysis on IMDB Movie Reviews

This project performs sentiment analysis on the IMDB movie reviews dataset using **Natural Language Processing (NLP)** and machine learning techniques. It includes text preprocessing, feature extraction, dimensionality reduction, and model training/evaluation.

---

## 1. Text Preprocessing

Raw text data is cleaned and transformed using standard NLP techniques:
- **Lowercasing:** Converts all characters to lowercase for consistency.
- **Regular Expressions:** Removes special characters and numbers.
- **Tokenization:** Splits the text into individual words (tokens).
- **Stopword Removal:** Eliminates common English words that add little semantic value (e.g., "the", "is", "and").
- **Lemmatization:** Reduces words to their base form (e.g., "running" → "run").

These steps are essential for cleaning the text and preparing it for feature extraction.

---

## 2. Text Vectorization (TF-IDF) and Dimensionality Reduction (PCA)

- **TF-IDF (Term Frequency-Inverse Document Frequency):** Converts the cleaned text into numerical vectors that reflect the importance of each word in a review, relative to the entire dataset. A maximum of 5,000 features is used.
- **PCA (Principal Component Analysis):** Reduces the high-dimensional TF-IDF vectors to 100 components, preserving most of the variance in the data. This improves computational efficiency while retaining important information.

---

## 3. Model Training

Two different classifiers are trained and evaluated on both the original and PCA-reduced datasets:
- **Support Vector Machine (SVM)** with a linear kernel.
- **Random Forest Classifier** with 100 estimators.

The dataset is split into training and testing sets (80% training / 20% testing) using `train_test_split`.

---

## 4. Model Performance

Both models demonstrate **very good performance** in classifying reviews as positive or negative.

### Results Overview:
- **SVM:** Performs strongly on both original and reduced data, showing robustness and efficiency.
- **Random Forest:** Also achieves high accuracy, particularly on the original dataset.

Evaluation metrics such as accuracy, confusion matrix, precision, recall, and F1-score confirm the effectiveness of both models. The PCA-reduced data retains nearly the same performance while significantly lowering feature dimensionality.

---

