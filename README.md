# 🍽️ Restaurant Rating Predictor Using NLP & Machine Learning

This repository presents a data science project focused on **predicting restaurant ratings** based on user-generated textual reviews. By leveraging **Natural Language Processing (NLP)** and **machine learning models**, this project aims to bridge the gap between qualitative customer feedback and quantitative business metrics.

The solution is built using a structured approach involving data cleaning, sentiment analysis, feature engineering with TF-IDF, and the application of supervised learning techniques. This work also supports an academic **research paper**, highlighting how AI can assist in automating customer feedback interpretation for restaurants, food delivery services, and review-based platforms.

---

## 🧠 Project Objectives

- Predict the **star rating (1 to 5)** for a restaurant based solely on the **written review**.
- Integrate **sentiment analysis** with **text vectorization** to enhance prediction accuracy.
- Compare the performance of different ML models and select the most robust one for deployment.
- Present a clean, interpretable, and production-ready solution for potential real-world use.

---

## 📂 Dataset Overview

- **Source**: Yelp Restaurant Reviews  
- **Format**: CSV file with columns including review text and star ratings.

| Column Name | Description                  |
|-------------|------------------------------|
| `Text`      | Raw customer review          |
| `Rating`    | Star rating (1 to 5)         |
| (Engineered)| Cleaned text, sentiment, etc.|

---

## 📊 Methodology

### 1. 🧹 Data Cleaning & Preprocessing
- Removed null and duplicate values.
- Performed text normalization:
  - Lowercasing, punctuation removal, stopword filtering.
  - Tokenization and lemmatization using `nltk`.

### 2. 💬 Sentiment Analysis
- Used `TextBlob` to extract:
  - **Polarity**: Ranges from -1 (negative) to 1 (positive).
  - **Subjectivity**: Ranges from 0 (objective) to 1 (subjective).
- These scores were used as **additional features** alongside the text data.

### 3. ✨ Feature Engineering
- **TF-IDF Vectorization**:
  - Limited to top 5,000 terms to reduce dimensionality.
  - Captures term importance relative to the review corpus.
- Combined TF-IDF vectors with sentiment scores into a unified feature matrix.

```python
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['Cleaned_Text']).toarray()
X = np.hstack((X_tfidf, df[['Polarity', 'Subjectivety']].values))
```

---

## 🤖 Model Development

### 🧪 Models Explored
  Initial experiments with simpler models were omitted in final pipeline.

- ✅ Final Model: SVC (RBF kernel) with BaggingClassifier

   - Chosen for its robustness and improved generalization

   - Handled non-linear decision boundaries well

   - Bagging added ensemble strength and reduced variance

```python
bagging_svm = BaggingClassifier(estimator=SVC(kernel='rbf'), n_estimators=10, random_state=42)
bagging_svm.fit(X_train, y_train)
```
### 📈 Evaluation Metrics
- Accuracy on rounded predictions

- Classification Report (Precision, Recall, F1-Score)

- Confusion Matrix for detailed class performance

### ✅ Results Summary
| Metric	                 | Value                               |
|--------------------------|-------------------------------------|
|Model	                   | SVC with Bagging                    |
|Vectorizer	               | TF-IDF                              |
|Accuracy (rounded)        | ~64%                                |
|Best Performing Classes	 | 1, 2 and 5 -star ratings            |
|Feature Contributions	   | Sentiment improved class separation |

Insight: Reviews with mixed sentiment often leaned toward 3-star predictions. Highly subjective or emotional reviews skewed predictions toward 1 or 5.

## 📄 Research Paper
📑 The complete research paper details the methodology, comparative model results, challenges, and future work.

## 🌐 Potential Applications
- Restaurant Owners: Gain sentiment-based feedback insights without waiting for numeric ratings.

- Review Platforms: Flag mismatched or suspicious reviews.

- Food Delivery Apps: Improve rating predictions during onboarding of new restaurants.

- Recommender Systems: Use predicted ratings to enhance personalized suggestions.
