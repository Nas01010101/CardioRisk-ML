# CardioRisk ML

Predicting heart failure mortality using ML, based on the Chicco & Jurman (2020) study.

## What this is

A small project exploring the Heart Failure Clinical Records dataset (299 patients). I tried to replicate the original study's findings and ended up noticing some issues with their methodology — mainly that the "time" variable causes data leakage.

## Main findings

- **Top predictors**: Serum creatinine and ejection fraction (matches the original paper)
- **Problem found**: The time variable (follow-up period) inflates performance — patients who die have shorter follow-up by definition
- **Realistic AUC**: ~0.75-0.82 without time, vs ~0.85+ with it

## Run it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Streamlit dashboard
streamlit run app.py

# Or CLI
python3 src/main.py --model all
```

## Structure

```
├── app.py          # Streamlit dashboard
├── src/            # Core code
├── data/           # Dataset
└── reports/        # Generated plots
```

## Reference

> Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics.

## Note

As always, we take ML analysis on small retrospective datasets with a boat of salt. This is a learning project, not clinical software.
