# CardioRisk ML

## Heart Failure Mortality Prediction — A Critical Replication

This project implements a rigorous machine learning pipeline for predicting survival in patients with heart failure, critically examining the methodology of the original Chicco & Jurman (2020) study.

The approach prioritizes **interpretability**, **calibration**, and **critical analysis** of methodological issues (particularly the time variable leakage problem) over pure predictive metrics.

## Academic Reference
> **Chicco, D., & Jurman, G. (2020).** Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*, 20(1), 16. [DOI: 10.1186/s12911-020-1023-5](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5)

## Key Findings

### ✓ Confirmed: Top Predictive Features
When excluding the problematic time variable, the top predictors are:
1. **Serum Creatinine** (0.299 importance)
2. **Ejection Fraction** (0.229 importance)
3. **Age** (0.115 importance)

This aligns with the original study's core claim.

### ⚠️ Critical Issue: Time Variable Leakage
The original study includes "time" (follow-up period) as a feature:
- Correlation with death: **-0.53**
- Deceased patients: mean 71 days follow-up
- Survivors: mean 158 days follow-up

This creates information leakage — patients who die have shorter follow-up *by definition*.

### Model Performance (Without Time Variable)

| Model               | Test AUC | CV AUC (5-fold) | Sensitivity | Specificity |
|---------------------|----------|-----------------|-------------|-------------|
| Logistic Regression | 0.820    | 0.733 ± 0.062   | 0.37        | 0.93        |
| Random Forest       | 0.792    | 0.790 ± 0.046   | 0.37        | 0.88        |
| XGBoost             | 0.739    | 0.756 ± 0.032   | 0.42        | 0.85        |

These are more realistic than the inflated results when time is included.

## Streamlit Dashboard

Run the interactive dashboard:
```bash
source venv/bin/activate
streamlit run app.py
```

Features:
- **Study Overview**: Dataset characteristics and reference study
- **Key Findings**: Comparison of my results vs. original study
- **Critical Analysis**: Methodological concerns (time leakage, sample size)
- **Literature Gaps**: What's missing from current research
- **Prediction Tool**: Individual risk assessment (educational only)

## Project Structure
```
├── app.py              # Streamlit dashboard
├── data/               # Heart Failure Clinical Records (UCI)
├── src/
│   ├── data_loader.py  # Data loading and cleaning
│   ├── models.py       # Model definitions and tuning
│   ├── evaluation.py   # Calibration and DCA
│   └── main.py         # CLI pipeline
├── reports/            # Generated figures
└── requirements.txt
```

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run CLI pipeline
export PYTHONPATH=$PYTHONPATH:.
python3 src/main.py --model all

# Or run Streamlit app
streamlit run app.py
```

## Disclaimer
As always, we take ML analysis on small retrospective datasets with a boat of salt — a 299-patient single-center cohort cannot reliably generalize, and the time variable leakage inflates reported performance. This project is for educational purposes only.
