# Twitter Sentiment Analysis

NLP pipeline to classify tweets into **Positive, Negative, Neutral or Irrelevant**, using a custom text preprocessor, TF-IDF features and classic machine learning models (Logistic Regression, Decision Tree, Random Forest and a majority vote).

## Dataset

[Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) (Kaggle). Each row has a tweet id, the entity it refers to (Borderlands, Amazon, Microsoft, etc.), the sentiment and the text.

| File | Rows |
|---|---|
| `twitter_training.csv` | 74,682 |
| `twitter_validation.csv` | 1,000 |

The CSV files are not included in the repo. Download them from Kaggle and place them next to the notebook.

## Pipeline

1. **Cleaning** – remove empty and duplicated tweets.
2. **Preprocessing** (`Preprocessor` class) – lowercase, remove URLs, punctuation, numbers, emojis and English stopwords (NLTK), then lemmatize with spaCy (`en_core_web_sm`).
3. **Features** – TF-IDF with 1 to 3-grams, 8,000 features and `min_df=10`, fitted only on the training set.
4. **Models** – Logistic Regression, Decision Tree, Random Forest and a majority vote of the three.

## Results

On the validation set (999 tweets after cleaning). The numbers and the notebook outputs come from the original run; the later fixes do not change them (same label encoding, same lemmas).

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.84 | 0.84 |
| Decision Tree | 0.91 | 0.91 |
| Random Forest | **0.95** | **0.95** |
| Majority vote | **0.95** | **0.95** |

**Note:** the training set contains several variations of the same tweet (rows that share the same tweet id), and validation tweets may be similar to training tweets. A split grouped by tweet id would give a more conservative estimate.

## How to run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Then open `twitter_sentiment_analysis.ipynb` in Jupyter and run all cells. The notebook saves the trained vectorizer, models and preprocessing objects with `joblib` (`*.model`, `*.data` and `*.path` files, ignored by git).

## Author

José Adrián Ochoa Sánchez – [GitHub](https://github.com/Adrian-8a) · [LinkedIn](https://www.linkedin.com/in/adrian-ochoa-ai)
