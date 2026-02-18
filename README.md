# CardioRisk ML

Predicting heart failure mortality using machine learning, based on the Chicco & Jurman (2020) study.

**Author:** Anas El-Ghoudane

## Overview

A critical replication of the original study using the Heart Failure Clinical Records dataset (299 patients). This project identifies methodological issues in the original paper — specifically that the "time" variable causes data leakage — and provides an honest assessment of what ML can realistically achieve on this data.

## Key Findings

- **Top predictors**: Serum creatinine and ejection fraction (confirms the original paper)
- **Critical issue**: The time variable (follow-up period) inflates performance — patients who die have shorter follow-up by definition
- **Realistic AUC**: ~0.75-0.82 without time leakage, vs ~0.85+ with it

## Features

The Streamlit dashboard includes:

- **Study Overview** — Dataset exploration and key statistics
- **Key Findings** — Model performance comparison and feature importance
- **Critical Analysis** — Detailed examination of data leakage and methodology issues
- **Project Limitations** — Academic context and limitations of current research
- **Statistical Analysis** — Distribution comparisons, hypothesis tests, and correlations with clear interpretations
- **Research Assistant 🔬** — RAG-powered Q&A over ~100 cardiovascular risk papers from PubMed Central

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py

# Or use CLI
python3 src/main.py --model all
```

### Research Assistant Setup

The Research Assistant page requires a Google Gemini API key (free tier available):

```bash
# Get a key at https://aistudio.google.com/apikey
export GOOGLE_API_KEY="your-key-here"
streamlit run app.py
```

On first use, click **"Build Knowledge Base"** to fetch and index ~100 papers from PubMed Central. This is a one-time setup (~5-10 minutes).

## Project Structure

```
├── app.py              # Streamlit dashboard
├── src/                # Core ML code
│   ├── main.py         # CLI entry point
│   ├── data_loader.py  # Data loading and preprocessing
│   ├── models.py       # Model training
│   ├── evaluation.py   # Evaluation metrics
│   └── rag/            # RAG research assistant
│       ├── paper_fetcher.py  # PubMed Central paper downloader
│       ├── vector_store.py   # FAISS vector store builder
│       └── query_engine.py   # Gemini-powered Q&A engine
├── r_analysis/         # R statistical analysis scripts
├── data/               # Heart failure dataset + papers + vector store
└── reports/            # Generated plots and analysis outputs
```

## Reference

> Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*.

## Disclaimer

As always, we take ML analysis on small retrospective datasets with a grain of salt — a 299-patient single-center cohort cannot reliably generalize. This is an educational project, not clinical software.
