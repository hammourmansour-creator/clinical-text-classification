Clinical Text Classification Mini-Pipeline (Python)

End-to-end NLP text classification project using the PubMed 20k RCT dataset.
Built with a research-oriented workflow:

EDA → Preprocessing → Classical Baseline → Transformer Model → Evaluation & Comparison

This repository is designed to be reproducible, interpretable, and interview-ready, demonstrating applied machine learning for healthcare text.

📌 Project Goal

Given a sentence from a medical abstract, predict its section label:

BACKGROUND

OBJECTIVE

METHODS

RESULTS

CONCLUSIONS

This mirrors real clinical NLP tasks such as structuring medical literature and supporting evidence retrieval systems.

🧠 Models Implemented
🔹 Classical Baseline — TF-IDF + Logistic Regression

Fast and interpretable

Strong benchmark for medical text

Captures vocabulary patterns and n-grams

🔹 Transformer Model — DistilBERT Fine-Tuning

Context-aware language understanding

Improved overall accuracy and class stability

Demonstrates modern NLP workflows

📊 Key Results (Dev Set)
Model	Accuracy	Objective F1	Notes
TF-IDF + LogReg	~83%	0.62	Strong baseline, fast, interpretable
DistilBERT	~87%	0.64	Better overall performance & stability
🔎 Insights

Transformers improved overall accuracy by ~4%.

Largest confusion remained between OBJECTIVE and BACKGROUND.

Baseline remained competitive due to structured medical vocabulary.

🗂 Repository Structure

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

🔄 Workflow
1️⃣ Exploratory Data Analysis

Verified dataset integrity and splits

Examined class balance and text length distributions

Confirmed consistent labels across splits

2️⃣ Preprocessing

Text normalization and cleaning

Label consistency checks

Prepared data for both classical ML and transformers

3️⃣ Baseline Model

TF-IDF feature extraction (unigrams + bigrams)

Logistic Regression classifier

Evaluated with accuracy, F1, and confusion matrix

4️⃣ Transformer Model

DistilBERT fine-tuning using Hugging Face

Tokenization and attention masking

GPU training (Google Colab)

5️⃣ Evaluation & Comparison

Per-class precision, recall, F1

Confusion matrix analysis

Baseline vs Transformer comparison tables

⚙️ Setup
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Download dataset
Download PubMed 20k RCT and place files in:
data/raw/pubmed_20k_rct/
    train.txt
    dev.txt
    test.txt
Dataset source: https://github.com/Franck-Dernoncourt/pubmed-rct

▶️ How to Run

Run notebooks in order:

01_eda.ipynb — dataset checks & exploration

02_preprocessing.ipynb — cleaning & preparation

03_baseline.ipynb — TF-IDF baseline

04_distilbert.ipynb — transformer training (Colab recommended)

05_comparison.ipynb — model comparison & exports

All outputs are saved in artifacts/.

📈 Evaluation Metrics

We evaluate using:

Accuracy

Precision / Recall / F1 (per class)

Macro vs Weighted averages

Confusion matrix analysis

Special focus: confusion between OBJECTIVE and BACKGROUND due to overlapping vocabulary.

🎯 Why This Project Matters

This project demonstrates:

Applied NLP for healthcare

End-to-end ML pipeline design

Model evaluation and error analysis

Baseline vs transformer trade-offs

Reproducible research workflow

Relevant to:

Clinical NLP research

Medical AI systems

Decision support tools

Production ML pipelines

🔮 Future Improvements

Class weighting for OBJECTIVE vs BACKGROUND

Biomedical transformers (BioBERT, PubMedBERT)

Hyperparameter tuning

Model deployment as an API

👤 Author

Mansour Hammour
Computer Science & Medical Student
Interested in AI for Healthcare, NLP, and Clinical Decision Support Systems

🔗 LinkedIn: https://www.linkedin.com/in/mansour-hammour-776666324

🔗 GitHub: https://github.com/hammourmansour-creator

📄 License

Educational and research use.

---

