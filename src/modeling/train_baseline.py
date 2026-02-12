import json
from pathlib import Path

from src.data_loader import load_pubmed_20k_rct  # your existing loader
from src.modeling.baseline import build_tfidf_logreg_baseline
from src.evaluation.metrics import evaluate_predictions

def main():
    data = load_pubmed_20k_rct("data/raw/pubmed_20k_rct")

    train_df = data["train"]
    dev_df = data["dev"]
    test_df = data["test"]

    X_train, y_train = train_df["text"].astype(str), train_df["label"].astype(str)
    X_dev, y_dev = dev_df["text"].astype(str), dev_df["label"].astype(str)
    X_test, y_test = test_df["text"].astype(str), test_df["label"].astype(str)

    labels = sorted(y_train.unique())

    model = build_tfidf_logreg_baseline()
    model.fit(X_train, y_train)

    dev_pred = model.predict(X_dev)
    dev_eval = evaluate_predictions(y_dev, dev_pred, labels)

    test_pred = model.predict(X_test)
    test_eval = evaluate_predictions(y_test, test_pred, labels)

    Path("results").mkdir(exist_ok=True)

    with open("results/dev_metrics.json", "w", encoding="utf-8") as f:
        json.dump(dev_eval.report | {"accuracy": dev_eval.accuracy}, f, indent=2)

    with open("results/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_eval.report | {"accuracy": test_eval.accuracy}, f, indent=2)

    dev_eval.confusion_df.to_csv("results/dev_confusion_matrix.csv")
    test_eval.confusion_df.to_csv("results/test_confusion_matrix.csv")

    print("Saved results to /results")
    print("Dev accuracy:", dev_eval.accuracy)
    print("Test accuracy:", test_eval.accuracy)

if __name__ == "__main__":
    main()

