from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_tfidf_logreg_baseline(
    ngram_range=(1, 2),
    max_features=50000,
    max_iter=2000,
) -> Pipeline:
    """
    Returns a strong classical NLP baseline:
      TF-IDF (turn text into numeric features) -> Logistic Regression (classifier)

    - ngram_range=(1,2) uses unigrams + bigrams (often improves medical text)
    - max_features limits vocabulary size for speed/memory
    - max_iter higher to ensure convergence
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram_range,
            max_features=max_features
        )),
        ("clf", LogisticRegression(max_iter=max_iter))
    ])
