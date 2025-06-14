# SciReX: An Integrated Framework for AI-Powered Scientific Reasoning

**SciReX** is a modular and scalable AI framework designed to automate scientific reasoning by integrating:
- Summarization (Extractive + Abstractive)
- Entity Extraction
- Fact Verification
- Research Gap Detection
- Paper Comparison & Trend Analysis

SciReX helps researchers efficiently analyze, compare, and verify scientific literature at scale.

---

## Features

- **Abstractive Summarization** using fine-tuned BART models
- **Entity & Relation Extraction** with SciBERT and DistilBERT
- **Fact Verification** via Wikidata & Semantic Scholar API
- **Topic Modeling** with BERTopic & LDA
- **Gap Analysis** using keyword-based scoring
- **PDF Parsing** and Real-time Paper Analysis
- **Interactive Streamlit UI** for end-to-end workflow

---

## Project Structure

```

SciReX/
├── app/                        # Streamlit app interface
├── models/                     # Fine-tuned models (BART, SciBERT, etc.)
├── data/                       # Datasets & annotations
│   ├── cord19/
│   ├── s2orc/
│   └── sciREX/
├── utils/                      # Preprocessing, visualization, helper functions
├── notebooks/                  # Jupyter notebooks for experiments
├── results/                    # Output of experiments (summaries, gaps, comparisons)
├── requirements.txt
└── README.md

````

---

##  Installation

```bash
git clone https://github.com/yourusername/sciREX.git
cd sciREX
pip install -r requirements.txt
````

---

## Datasets Used

* [S2ORC](https://allenai.org/data/s2orc)
* [CORD-19](https://www.semanticscholar.org/cord19)
* [SciREX Dataset](https://allenai.org/data/scirex)
* [Wikidata](https://www.wikidata.org/)
* [Semantic Scholar API](https://api.semanticscholar.org/)

---

##  How to Use

### 1. Start the App

```bash
streamlit run app/main.py
```

###  Upload a Paper (PDF or Abstract Text)

###  Choose Options:

* Generate Summary
* Extract Entities
* Check Factual Claims
* Compare Papers
* Detect Research Gaps

---

##  Models

* `facebook/bart-large-cnn` (Summarization)
* `allenai/scibert-scivocab-uncased` (NER, RE)
* `sentence-transformers/all-MiniLM-L6-v2` (Semantic similarity)
* BERTopic + UMAP + HDBSCAN (Topic modeling)

---

## Evaluation Metrics

* **Summarization**: ROUGE-1, ROUGE-2, ROUGE-L
* **NER**: Precision, Recall, F1-Score
* **Fact Verification**: Accuracy (on annotated claim dataset)
* **Gap Detection**: Custom scoring and coverage metrics
* **Paper Comparison**: Cosine similarity, Method/Metric/Reference overlap

---

##  Sample Results

| Component     | Metric           | Score |
| ------------- | ---------------- | ----- |
| Summarization | ROUGE-L          | 0.45  |
| NER           | F1-Score         | 0.81  |
| Fact Checking | Accuracy         | 87%   |
| Gap Detection | Topical Coverage | 92%   |

---

##  Limitations

* Focused on AI/ML domain (generalization may be limited)
* PDF parsing may struggle with complex formatting
* Rule-based extraction may miss unconventional patterns

---

##  Future Work

* Add multilingual support
* Integrate citation network analysis
* Improve layout-aware PDF parsing
* Dynamic, self-updating research gap detection
* More robust evaluation framework for factual consistency

---

##  Contributors

* **Nishit Bohra** — Framework Design & Lead
* **Omkar H. Thorve** — NER & Fact Verification
* **Avantika R. Patil** — Dataset Curation & Summarization Models
* **Dr. Aniket Shahade** — Research Supervision & Methodology

---

##  License

MIT License. See `LICENSE` file for details.

---

##  Acknowledgements

* Allen Institute for AI for CORD-19 and S2ORC datasets
* Hugging Face Transformers
* Semantic Scholar & Wikidata APIs

```
