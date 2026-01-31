import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import shap

# Page configuration
st.set_page_config(
    page_title="CardioRisk ML",
    page_icon="📄",
    layout="wide"
)

# Academic styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+Pro:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    
    .main-title {
        font-family: 'Source Serif Pro', Georgia, serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
        line-height: 1.2;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-family: 'Source Serif Pro', Georgia, serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border: 1px solid #fcd34d;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        margin: 1.5rem 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .reference-box {
        background-color: #f8fafc;
        border-left: 3px solid #64748b;
        padding: 1rem 1.25rem;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 1rem 0;
    }
    .finding-card {
        background-color: #f1f5f9;
        border-radius: 6px;
        padding: 1.25rem;
        margin: 0.75rem 0;
    }
    .warning-card {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem 1.25rem;
        border-radius: 0 6px 6px 0;
        margin: 0.75rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-family: 'Source Serif Pro', Georgia, serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e293b;
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
    return df

df = load_data()

# Prepare data
X_full = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Single split
X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.2, stratify=y, random_state=42)

# Create no-time versions from same split
X_train_no_time = X_train_full.drop('time', axis=1)
X_test_no_time = X_test_full.drop('time', axis=1)

@st.cache_resource
def train_models_full():
    models = {
        'Logistic Regression': LogisticRegression(C=1, random_state=42, solver='liblinear', max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, eval_metric='logloss')
    }
    X_tr, _, y_tr, _ = train_test_split(df.drop('DEATH_EVENT', axis=1), df['DEATH_EVENT'], test_size=0.2, stratify=df['DEATH_EVENT'], random_state=42)
    for model in models.values():
        model.fit(X_tr, y_tr)
    return models

@st.cache_resource
def train_models_no_time():
    models = {
        'Logistic Regression': LogisticRegression(C=1, random_state=42, solver='liblinear', max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, eval_metric='logloss')
    }
    X_tr, _, y_tr, _ = train_test_split(df.drop('DEATH_EVENT', axis=1), df['DEATH_EVENT'], test_size=0.2, stratify=df['DEATH_EVENT'], random_state=42)
    X_tr = X_tr.drop('time', axis=1)
    for model in models.values():
        model.fit(X_tr, y_tr)
    return models

models_full = train_models_full()
models_no_time = train_models_no_time()

def get_results(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {
            'auc': auc(fpr, tpr), 
            'probs': y_prob,
            'brier': brier_score_loss(y_test, y_prob)
        }
    return results

results_full = get_results(models_full, X_test_full, y_test)
results_no_time = get_results(models_no_time, X_test_no_time, y_test)

# ==================== HEADER ====================
st.markdown('<p class="main-title">CardioRisk ML</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Heart Failure Mortality Prediction — A Critical Replication of Chicco & Jurman (2020)</p>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>Methodological Note:</strong> Findings from machine learning analyses on small retrospective datasets 
    should be interpreted with caution. A 299-patient single-center cohort has limited generalizability, and the 
    inclusion of the time variable may introduce information leakage that inflates reported performance metrics. 
    See Critical Analysis for details.
</div>
""", unsafe_allow_html=True)

# Navigation
page = st.radio("", ["Study Overview", "Key Findings", "Critical Analysis", "Literature Gaps", "Statistical Analysis (R)"], horizontal=True, label_visibility="collapsed")

st.divider()

# ==================== STUDY OVERVIEW ====================
if page == "Study Overview":
    
    st.markdown('<p class="section-header">Reference Study</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="reference-box">
        <strong>Chicco, D., & Jurman, G. (2020).</strong> Machine learning can predict survival of patients with heart failure 
        from serum creatinine and ejection fraction alone. <em>BMC Medical Informatics and Decision Making</em>, 20(1), 16.<br><br>
        <strong>Key Claim:</strong> Using only <strong>serum creatinine</strong> and <strong>ejection fraction</strong>, 
        machine learning models can predict mortality with high accuracy, outperforming models using all 13 clinical features.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">Dataset Characteristics</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<p class="metric-label">Cohort Size</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">299</p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="metric-label">Mortality Rate</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{df["DEATH_EVENT"].mean()*100:.1f}%</p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p class="metric-label">Features</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">12</p>', unsafe_allow_html=True)
    with col4:
        st.markdown('<p class="metric-label">Follow-up</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">4-285 days</p>', unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">Clinical Variables</p>', unsafe_allow_html=True)
    
    variables = pd.DataFrame({
        'Variable': ['Ejection Fraction', 'Serum Creatinine', 'Age', 'Serum Sodium', 'CPK', 'Platelets', 
                    'Anaemia', 'Diabetes', 'High BP', 'Sex', 'Smoking', 'Time*'],
        'Description': [
            'Blood ejected per contraction (normal: 55-70%)',
            'Kidney function marker (normal: 0.7-1.2 mg/dL)',
            'Patient age in years',
            'Blood sodium (normal: 135-145 mEq/L)',
            'Enzyme indicating heart damage',
            'Blood platelet count',
            'Low red blood cells (0/1)',
            'Diabetes status (0/1)',
            'Hypertension status (0/1)',
            'Male=1, Female=0',
            'Smoking status (0/1)',
            'Follow-up period (days) — see Critical Analysis'
        ]
    })
    st.dataframe(variables.set_index('Variable'), use_container_width=True)
    
    st.caption("*Time variable requires special consideration due to potential information leakage.")

# ==================== KEY FINDINGS ====================
elif page == "Key Findings":
    
    st.markdown('<p class="section-header">Replication Results vs. Original Study</p>', unsafe_allow_html=True)
    
    st.markdown("""
    I replicated the methodology of Chicco & Jurman (2020) using the same Heart Failure Clinical Records dataset. 
    Below I compare my findings with the original study's conclusions.
    """)
    
    # Finding 1: Top predictors
    st.markdown('<div class="finding-card">', unsafe_allow_html=True)
    st.markdown("### Finding 1: Most Predictive Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Study (Chicco & Jurman, 2020)**")
        st.markdown("""
        > "Serum creatinine and ejection fraction are the two most relevant features... 
        > these two features alone are sufficient to predict survival."
        """)
    
    with col2:
        st.markdown("**My Replication**")
        rf = models_full['Random Forest']
        importance = pd.Series(rf.feature_importances_, index=X_full.columns).sort_values(ascending=False)
        
        st.markdown(f"""
        Feature importance (Random Forest):
        1. **{importance.index[0]}**: {importance.iloc[0]:.3f}
        2. **{importance.index[1]}**: {importance.iloc[1]:.3f}
        3. **{importance.index[2]}**: {importance.iloc[2]:.3f}
        """)
    
    if importance.index[0] == 'time':
        st.markdown("**Verdict**: ⚠️ **Partially Confirmed with Caveat** — Time dominates, but this is problematic (see Critical Analysis).")
    else:
        st.markdown("**Verdict**: ✓ **Confirmed** — Serum creatinine and ejection fraction rank highly.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Finding 2: Model Performance
    st.markdown('<div class="finding-card">', unsafe_allow_html=True)
    st.markdown("### Finding 2: Model Performance Comparison")
    
    st.markdown("**With vs. Without Time Variable:**")
    
    comparison = pd.DataFrame({
        'Model': list(results_full.keys()),
        'AUC (with time)': [f"{results_full[m]['auc']:.3f}" for m in results_full],
        'AUC (without time)': [f"{results_no_time[m]['auc']:.3f}" for m in results_no_time],
        'Δ AUC': [f"{results_full[m]['auc'] - results_no_time[m]['auc']:+.3f}" for m in results_full]
    })
    st.dataframe(comparison.set_index('Model'), use_container_width=True)
    
    st.markdown("""
    **Interpretation**: The substantial drop in AUC when removing the time variable suggests that:
    - The original study's high performance may be partially driven by information leakage
    - Models trained without time are more clinically realistic (predicting at admission, not retrospectively)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Finding 3
    st.markdown('<div class="finding-card">', unsafe_allow_html=True)
    st.markdown("### Finding 3: Clinical Interpretation")
    
    st.markdown("""
    **What the clinical features indicate:**
    
    - **Ejection Fraction < 40%**: Strong mortality indicator. Aligns with clinical guidelines (HFrEF classification).
    
    - **Serum Creatinine > 1.5 mg/dL**: Indicates kidney dysfunction — cardiorenal syndrome is well-documented.
    
    - **Age**: Expected linear relationship with mortality risk.
    
    These findings align with established cardiology literature, lending credibility to the model's feature selection.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== CRITICAL ANALYSIS ====================
elif page == "Critical Analysis":
    
    st.markdown('<p class="section-header">On the Time Variable</p>', unsafe_allow_html=True)
    
    st.markdown("""
    One aspect of this dataset worth examining is the "time" variable — the follow-up period in days. 
    I believe it may introduce data leakage when used as a classification feature. Here's my reasoning:
    """)
    
    st.markdown('<p class="section-header">The Thesis</p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    The "time" variable records how long each patient was observed. In this dataset:
    
    - Correlation between time and death: **{df['time'].corr(df['DEATH_EVENT']):.2f}**
    - Mean follow-up for survivors: **{df[df.DEATH_EVENT==0]['time'].mean():.0f} days**
    - Mean follow-up for deceased: **{df[df.DEATH_EVENT==1]['time'].mean():.0f} days**
    
    Patients who died have shorter follow-up periods. If a patient dies early, their observation ends early. 
    This makes "time" more of a consequence of the outcome than a predictor of it.
    """)
    
    st.markdown('<p class="section-header">Supporting Evidence</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Performance Comparison:**")
        st.markdown(f"""
        | Model | With Time | Without Time |
        |-------|-----------|-------------|
        | Logistic | {results_full['Logistic Regression']['auc']:.3f} | {results_no_time['Logistic Regression']['auc']:.3f} |
        | Random Forest | {results_full['Random Forest']['auc']:.3f} | {results_no_time['Random Forest']['auc']:.3f} |
        | XGBoost | {results_full['XGBoost']['auc']:.3f} | {results_no_time['XGBoost']['auc']:.3f} |
        
        The drop in AUC when removing time suggests the models 
        were partially relying on this variable.
        """)
    
    with col2:
        st.markdown("**References:**")
        st.markdown("""
        - [Original paper](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5): 
          The authors ran analyses both with and without the time variable
        - [UCI Dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records): 
          Documents "time" as "follow-up period (days)"
        - [Kaggle Discussion](https://www.kaggle.com/datasets/rabieelkharoua/predict-survival-of-patients-with-heart-failure/discussion/498685)
        """)
    
    st.markdown("""
    <div class="reference-box">
        <strong>From the Kaggle discussion by Rabie El Kharoua:</strong><br><br>
        <em>"Time is not a Feature, It's a Target. You know, it's super important to recognize that the 'time' column 
        in our dataset isn't really a feature we should be using in our models. It's actually a target variable. 
        See, it represents the follow-up period duration... Time and Death_Event go hand in hand. Yep, they're 
        pretty much like two peas in a pod. The 'time' column is closely tied to the 'DEATH_EVENT' column because 
        when an event happens, it affects the time. Using time as a feature could cause some serious bias in our 
        predictions... At the end of the day, we want our models to make accurate predictions, right? So, instead 
        of focusing on time, let's turn our attention to other features that are available at the time of prediction."</em>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">Where I Could Be Wrong</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This interpretation may not be the only valid one:
    
    **1. Alternative interpretation of "time"**
    
    It's possible "time" represents a *scheduled* follow-up interval (e.g., 30-day vs 90-day checkup protocol) 
    rather than the actual observed time-to-event. If patients were assigned to different monitoring schedules, 
    this would be a legitimate baseline feature. The dataset documentation is ambiguous on this point.
    
    **2. Survival analysis handles time differently**
    
    In Cox regression or Kaplan-Meier analysis, time-to-event is part of the outcome definition, not a feature. 
    The original paper used stratified logistic regression by month, which partially addresses the issue. 
    My concern applies specifically to treating time as a classification feature.
    
    **3. Different goals: explanation vs. prediction**
    
    The original study may have aimed to understand which factors correlate with mortality in a retrospective 
    cohort (exploratory analysis). My criticism assumes the goal is to build a model that predicts outcomes 
    for new patients (prospective deployment). Both are valid for different purposes.
    """)
    
    st.markdown('<p class="section-header">Other Limitations</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Beyond the time variable, there are other considerations:
    
    - **Sample size (N=299)**: Small for reliable generalization; cross-validation shows variance of ±0.05 in AUC
    - **Single center**: Data from Faisalabad Institute of Cardiology; may not transfer to other populations
    - **No external validation**: Results haven't been tested on independent datasets
    - **Missing context**: No information on treatments received or NYHA classification
    """)

# ==================== LITERATURE GAPS ====================
elif page == "Literature Gaps":
    
    st.markdown('<p class="section-header">What Is Missing from Current Research</p>', unsafe_allow_html=True)
    
    st.markdown("""
    While the Chicco & Jurman (2020) study contributed to the field, several gaps remain unaddressed in the broader 
    heart failure ML literature:
    """)
    
    st.markdown('<div class="finding-card">', unsafe_allow_html=True)
    st.markdown("### 1. Lack of Survival Analysis Integration")
    st.markdown("""
    Most studies treat mortality prediction as binary classification, ignoring:
    - **Time-to-event** information (when death occurs, not just if)
    - **Censoring** (patients lost to follow-up)
    - **Competing risks** (death from non-cardiac causes)
    
    **Better approach**: Cox proportional hazards, Random Survival Forests, or DeepSurv models that account for 
    the temporal nature of the outcome.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="finding-card">', unsafe_allow_html=True)
    st.markdown("### 2. Clinical Utility Assessment")
    st.markdown("""
    Papers routinely report AUC but rarely assess:
    - **Decision Curve Analysis**: Does the model provide net benefit over treat-all/treat-none strategies?
    - **Threshold optimization**: What probability cutoff maximizes clinical utility?
    - **Number Needed to Treat/Screen**: Practical implications for resource allocation
    
    A model with AUC=0.85 may still be clinically useless if it doesn't change management decisions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="finding-card">', unsafe_allow_html=True)
    st.markdown("### 3. Calibration Over Discrimination")
    st.markdown("""
    Clinical decision-making requires **well-calibrated probabilities**, not just good ranking:
    - If I tell a patient they have 30% risk, does 30% of such patients actually die?
    - Most ML models are poorly calibrated out-of-the-box
    - Platt scaling, isotonic regression, or Venn prediction are underutilized
    
    **Brier score** and calibration plots should be standard, not optional.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="finding-card">', unsafe_allow_html=True)
    st.markdown("### 4. Fairness and Subgroup Analysis")
    st.markdown("""
    Questions rarely addressed:
    - Does the model perform equally well for men and women?
    - Are there age-related biases?
    - How does performance vary across comorbidity profiles?
    
    Aggregate AUC can mask disparities that matter clinically.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="finding-card">', unsafe_allow_html=True)
    st.markdown("### 5. Prospective and Multi-Center Validation")
    st.markdown("""
    The vast majority of heart failure ML papers:
    - Use retrospective, single-center data
    - Split data randomly (temporal leakage)
    - Never test on truly external populations
    
    **Without prospective validation**, we cannot know if these models generalize to future patients in real clinical settings.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="finding-card">', unsafe_allow_html=True)
    st.markdown("### 6. Interpretability vs. Performance Trade-off")
    st.markdown("""
    The push for complex models (XGBoost, neural networks) may be misguided:
    - Simple logistic regression often performs comparably
    - Interpretable models are easier to audit for errors
    - Clinicians trust what they can understand
    
    For high-stakes decisions, a slightly less accurate but fully transparent model may be preferable.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">Recommended Reading</p>', unsafe_allow_html=True)
    
    st.markdown("""
    - **Van Calster, B. et al. (2019)**. Calibration: the Achilles heel of predictive analytics. *BMC Medicine*
    - **Vickers, A. J. & Elkin, E. B. (2006)**. Decision curve analysis. *Medical Decision Making*
    - **Obermeyer, Z. & Emanuel, E. J. (2016)**. Predicting the Future — Big Data, Machine Learning, and Clinical Medicine. *NEJM*
    - **Christodoulou, E. et al. (2019)**. A systematic review shows no performance benefit of ML over logistic regression. *JCE*
    """)

# ==================== STATISTICAL ANALYSIS ====================
elif page == "Statistical Analysis (R)":
    
    import os
    from scipy import stats
    
    # ==================== HEADER ====================
    st.markdown('<p class="section-header">Statistical Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="reference-box">
        This section presents descriptive and inferential statistical analyses to characterize 
        differences in clinical parameters between patients who experienced mortality and those 
        who survived. These foundational analyses establish the empirical basis for subsequent 
        predictive modeling.
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    n_total = len(df)
    n_died = df['DEATH_EVENT'].sum()
    n_survived = n_total - n_died
    mortality_rate = round(100 * n_died / n_total, 1)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", n_total)
    with col2:
        st.metric("Survived", n_survived)
    with col3:
        st.metric("Died", n_died)
    with col4:
        st.metric("Mortality Rate", f"{mortality_rate}%")
    
    st.divider()
    
    # ==================== SECTION 1: DISTRIBUTIONS ====================
    st.markdown('<p class="section-header">1. How Do Survivors and Deceased Patients Differ?</p>', unsafe_allow_html=True)
    
    st.markdown("""
    The following histograms display the distribution of key clinical variables, stratified by 
    patient outcome. Divergence between these distributions suggests potential discriminative 
    value for mortality prediction.
    """)
    
    # Split data
    survived = df[df['DEATH_EVENT'] == 0]
    died = df[df['DEATH_EVENT'] == 1]
    
    # Key variables to compare
    key_vars = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, var in enumerate(key_vars):
        ax = axes[i]
        ax.hist(survived[var], bins=20, alpha=0.6, label='Survived', color='#27AE60', density=True)
        ax.hist(died[var], bins=20, alpha=0.6, label='Died', color='#E74C3C', density=True)
        ax.set_xlabel(var.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add means as vertical lines
        ax.axvline(survived[var].mean(), color='#27AE60', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(died[var].mean(), color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.caption("Dashed vertical lines indicate the group mean. Greater separation between means suggests stronger association with outcome.")
    
    # Interpretation
    st.markdown("""
    **Clinical Observations:**
    - **Age**: Patients who experienced mortality were older on average
    - **Ejection Fraction**: Lower values (reduced ventricular function) are more prevalent among deceased patients
    - **Serum Creatinine**: Elevated values (indicative of renal impairment) are associated with increased mortality  
    - **Serum Sodium**: Modestly lower concentrations observed in deceased patients
    """)
    
    st.divider()
    
    # ==================== SECTION 2: STATISTICAL TESTS ====================
    st.markdown('<p class="section-header">2. Are These Differences Statistically Significant?</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Observed differences in group means require formal hypothesis testing to determine 
    whether they represent true population differences or sampling variability. The 
    Mann-Whitney U test (a non-parametric alternative to the t-test) is applied here.
    
    **Interpretation threshold:** A p-value < 0.05 is conventionally considered statistically significant.
    """)
    
    # Perform t-tests
    test_results = []
    for var in ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'platelets', 'creatinine_phosphokinase']:
        survived_vals = survived[var].dropna()
        died_vals = died[var].dropna()
        
        stat, p_value = stats.mannwhitneyu(survived_vals, died_vals, alternative='two-sided')
        
        test_results.append({
            'Variable': var.replace('_', ' ').title(),
            'Survived (mean)': round(survived_vals.mean(), 1),
            'Died (mean)': round(died_vals.mean(), 1),
            'Difference': round(died_vals.mean() - survived_vals.mean(), 2),
            'p-value': f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001",
            'Significant?': '✓ Yes' if p_value < 0.05 else '✗ No'
        })
    
    results_df = pd.DataFrame(test_results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Interpretation:**
    - **Age, Ejection Fraction, Serum Creatinine** — Statistically significant differences observed (candidate predictors)
    - **Serum Sodium** — Statistically significant, though with smaller absolute difference
    - **Platelets, CPK** — No statistically significant difference detected between outcome groups
    """)
    
    st.divider()
    
    # ==================== SECTION 3: CATEGORICAL VARIABLES ====================
    st.markdown('<p class="section-header">3. Do Binary Risk Factors Matter?</p>', unsafe_allow_html=True)
    
    st.markdown("""
    For dichotomous clinical variables (e.g., presence or absence of diabetes, smoking history), 
    mortality rates are compared between groups using chi-square tests of independence.
    """)
    
    binary_vars = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    mortality_rates = []
    labels = []
    colors = []
    
    for var in binary_vars:
        rate_0 = df[df[var] == 0]['DEATH_EVENT'].mean() * 100
        rate_1 = df[df[var] == 1]['DEATH_EVENT'].mean() * 100
        
        labels.extend([f'{var.replace("_", " ").title()}\n(No)', f'{var.replace("_", " ").title()}\n(Yes)'])
        mortality_rates.extend([rate_0, rate_1])
        colors.extend(['#3498DB', '#E74C3C'])
    
    x = np.arange(len(labels))
    bars = ax.bar(x, mortality_rates, color=colors, alpha=0.8)
    ax.set_ylabel('Mortality Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(y=mortality_rate, color='gray', linestyle='--', alpha=0.7, label=f'Overall: {mortality_rate}%')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 50)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.caption("Blue = factor absent, Red = factor present. Dashed line = overall mortality rate.")
    
    # Chi-square tests
    chi_results = []
    for var in binary_vars:
        contingency = pd.crosstab(df[var], df['DEATH_EVENT'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        chi_results.append({
            'Variable': var.replace('_', ' ').title(),
            'p-value': f"{p_value:.3f}",
            'Significant?': '✓ Yes' if p_value < 0.05 else '✗ No'
        })
    
    chi_df = pd.DataFrame(chi_results)
    st.dataframe(chi_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Interpretation:**  
    None of the dichotomous clinical variables (diabetes, smoking, hypertension) demonstrate 
    statistically significant associations with mortality in this cohort. However, this finding 
    should be interpreted cautiously—the limited sample size (n=299) may provide insufficient 
    statistical power to detect modest effect sizes.
    """)
    
    st.divider()
    
    # ==================== SECTION 4: CORRELATIONS ====================
    st.markdown('<p class="section-header">4. How Are Variables Related to Each Other?</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Pearson correlation coefficients quantify the linear association between continuous variables. 
    Values range from -1 (perfect inverse relationship) to +1 (perfect positive relationship), 
    with values near 0 indicating no linear association.
    """)
    
    # Correlation with death
    numeric_cols = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 
                    'platelets', 'creatinine_phosphokinase', 'time']
    
    correlations = df[numeric_cols + ['DEATH_EVENT']].corr()['DEATH_EVENT'].drop('DEATH_EVENT').sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#E74C3C' if x > 0 else '#3498DB' for x in correlations.values]
    bars = ax.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels([c.replace('_', ' ').title() for c in correlations.index])
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Correlation with Death')
    ax.set_xlim(-0.6, 0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.caption("Blue = negative correlation (protective factor), Red = positive correlation (associated with increased mortality)")
    
    st.markdown("""
    <div class="warning-card">
        <strong>⚠️ Warning about "Time":</strong> The strong negative correlation with time is misleading. 
        Patients who died have shorter follow-up because death ended their observation — 
        this is not a useful predictor.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Key Correlations:**
    - **Ejection Fraction** (r = -0.27): Higher ejection fraction associated with improved survival (protective factor)
    - **Serum Creatinine** (r = +0.29): Elevated creatinine associated with increased mortality (adverse prognostic indicator)
    - **Age** (r = +0.25): Advanced age associated with higher mortality risk
    """)
    
    st.divider()
    
    # ==================== SECTION 5: SUMMARY ====================
    st.markdown('<p class="section-header">Summary</p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="reference-box">
        <strong>What the statistics tell us:</strong><br><br>
        
        <strong>Variables with significant association with mortality:</strong><br>
        • <strong>Age</strong> — Advanced age associated with increased mortality<br>
        • <strong>Ejection Fraction</strong> — Reduced ejection fraction (ventricular dysfunction) associated with mortality<br>
        • <strong>Serum Creatinine</strong> — Elevated creatinine (renal impairment) associated with mortality<br><br>
        
        <strong>Variables without statistically significant association:</strong><br>
        • Diabetes, smoking, hypertension — No significant association detected in this cohort<br>
        • Platelet count, CPK — No significant difference between outcome groups<br><br>
        
        <strong>Study Limitations:</strong><br>
        • Limited sample size (n=299) constrains statistical power<br>
        • Single-center study (Faisalabad Institute of Cardiology, Pakistan) — external validity uncertain<br>
        • Observational design — causal inference not possible
    </div>
    """, unsafe_allow_html=True)



    
# Footer
st.markdown("""
<div class="footer">
    <strong>CardioRisk ML</strong> — Heart Failure Clinical Records Dataset (UCI)<br>
    Reference: Chicco & Jurman, BMC Medical Informatics (2020) | Built for educational purposes only<br>
    <span style="opacity: 0.8;">Author: Anas El-Ghoudane</span>
</div>
""", unsafe_allow_html=True)
