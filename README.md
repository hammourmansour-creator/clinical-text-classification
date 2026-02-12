# Clinical Text Classification Mini-Pipeline (Python)

End-to-end NLP text classification project using the **PubMed 20k RCT** dataset.  
Built with a research-oriented workflow: **EDA → preprocessing → classical baseline → transformer model → evaluation + comparison**.

This repo is designed to be understandable, reproducible, and interview-ready.

---

## What this project does

Given a sentence from a medical abstract, the model predicts its section label:

- **BACKGROUND**
- **OBJECTIVE**
- **METHODS**
- **RESULTS**
- **CONCLUSIONS**

Two model families are implemented and compared:

1. **Baseline**: TF-IDF + Logistic Regression (fast, strong classical NLP baseline)
2. **Transformer**: DistilBERT fine-tuning (context-aware model)

---

## Repository structure

.
├── artifacts/
│ ├── baseline/
│ │ ├── metrics_dev.json
│ │ └── confusion_matrix_dev.csv
│ ├── distilbert/
│ │ ├── metrics_dev.json
│ │ └── confusion_matrix_dev.csv
│ └── comparison/
│ ├── overall_metrics_dev.csv
│ ├── per_class_metrics_dev.csv
│ ├── confusion_delta_dev.csv
│ └── notes_summary.txt
│
├── data/
│ └── raw/
│ └── pubmed_20k_rct/
│ ├── train.txt
│ ├── dev.txt
│ └── test.txt
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_baseline.ipynb
│ ├── 04_distilbert.ipynb
│ └── 05_comparison.ipynb
│
├── src/
│ ├── data/
│ │ ├── data_loader.py
│ │ └── preprocessing.py
│ ├── modeling/
│ │ ├── baseline.py
│ │ ├── train_baseline.py
│ │ ├── trainer_weighted.py
│ │ └── transformer.py
│ ├── evaluation/
│ │ ├── metrics.py
│ │ ├── export.py
│ │ └── plot.py
│ └── utils/
│ └── io.py
│
├── .gitignore
├── requirements.txt
└── README.md



---

## Setup

1) Install dependencies
pip install -r requirements.txt

2) Download NLTK resources (run once)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); n

3) Add the dataset

Place the PubMed 20k RCT files here:

data/raw/pubmed_20k_rct/train.txt
data/raw/pubmed_20k_rct/dev.txt
data/raw/pubmed_20k_rct/test.txt



## How to run (recommended)

Open and run the notebooks in order:

01_eda.ipynb — sanity checks, label distribution, text-length EDA

02_preprocessing.ipynb — cleaning steps + label mapping checks

03_baseline.ipynb — TF-IDF + Logistic Regression training + evaluation

04_distilbert.ipynb — DistilBERT training + evaluation (recommended to run in Google Colab with GPU)

05_comparison.ipynb — compares both models and exports comparison tables

All key results are saved to artifacts/.


## Evaluation approach

For each model we compute:

Accuracy

Per-class precision / recall / F1

Macro avg vs Weighted avg

Confusion matrix (to study label confusion patterns)

A specific focus is the confusion between OBJECTIVE and BACKGROUND, because these sections can share overlapping vocabulary.

## Notes on baseline vs transformers

- TF-IDF + Logistic Regression is fast, interpretable, and strong for structured scientific text.
- DistilBERT captures contextual meaning and improves overall accuracy.
- Transformers improve high-frequency classes but may not fully resolve OBJECTIVE vs BACKGROUND confusion.
- This comparison highlights trade-offs between performance, cost, and interpretability.


## Author

Mansour Hammour
Computer Science & Medical student — building research-oriented ML/NLP projects for Healthcare AI.

LinkedIn: https://www.linkedin.com/in/mansour-hammour-776666324


---

## Final push checklist (do this once, then push)
1) Ensure dataset files are **NOT** committed (add to `.gitignore`):
```txt
data/raw/pubmed_20k_rct/

