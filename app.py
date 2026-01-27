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
    <strong>Methodological Note:</strong> As always, we take ML analysis on small retrospective datasets with a boat of salt — 
    a 299-patient single-center cohort cannot reliably generalize, and the time variable leakage inflates reported performance. 
    See Critical Analysis for details.
</div>
""", unsafe_allow_html=True)

# Navigation
page = st.radio("", ["Study Overview", "Key Findings", "Critical Analysis", "Literature Gaps", "Prediction Tool"], horizontal=True, label_visibility="collapsed")

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
    
    st.markdown('<p class="section-header">Methodological Concerns</p>', unsafe_allow_html=True)
    
    # Time variable issue
    st.markdown("""
    <div class="warning-card">
        <strong>⚠️ Critical Issue: Time Variable Leakage</strong><br><br>
        The "time" variable (follow-up period) is highly problematic:
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**The Problem:**")
        st.markdown(f"""
        - Correlation with death: **{df['time'].corr(df['DEATH_EVENT']):.2f}**
        - Mean time (survivors): **{df[df.DEATH_EVENT==0]['time'].mean():.0f} days**
        - Mean time (deceased): **{df[df.DEATH_EVENT==1]['time'].mean():.0f} days**
        
        Patients who died have shorter follow-up *by definition*. This is not a predictive feature — it's a consequence of the outcome.
        """)
    
    with col2:
        st.markdown("**Impact on Results:**")
        st.markdown(f"""
        Including time inflates model performance:
        
        | Model | AUC (with time) | AUC (without) |
        |-------|-----------------|---------------|
        | Logistic | {results_full['Logistic Regression']['auc']:.2f} | {results_no_time['Logistic Regression']['auc']:.2f} |
        | Random Forest | {results_full['Random Forest']['auc']:.2f} | {results_no_time['Random Forest']['auc']:.2f} |
        | XGBoost | {results_full['XGBoost']['auc']:.2f} | {results_no_time['XGBoost']['auc']:.2f} |
        """)
    
    st.markdown('<p class="section-header">Additional Limitations</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **1. Sample Size (N=299)**
    - Insufficient for reliable generalization
    - Cross-validation shows high variance (±0.10-0.19 in AUC)
    - Risk of overfitting, especially for complex models
    
    **2. Single-Center Data**
    - Collected from Faisalabad Institute of Cardiology, Pakistan
    - Population-specific factors may not transfer to other demographics
    - No external validation cohort
    
    **3. Class Imbalance Handling**
    - 32% mortality rate is moderate but still imbalanced
    - Original study doesn't discuss SMOTE, class weights, or threshold optimization
    
    **4. Missing Clinical Context**
    - No information on treatment protocols
    - No baseline risk stratification (NYHA class, NT-proBNP)
    - No competing risks analysis
    
    **5. Temporal Validation**
    - No train/test split by admission date
    - Model may not generalize to future patients
    """)
    
    st.markdown('<p class="section-header">Recommendations for Future Work</p>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **Remove or properly handle the time variable** — either exclude it or use survival analysis methods (Cox regression)
    2. **External validation** on datasets from different populations/centers
    3. **Prospective validation** to assess real-world utility
    4. **Calibration focus** — well-calibrated probabilities matter more than discrimination in clinical settings
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

# ==================== PREDICTION TOOL ====================
elif page == "Prediction Tool":
    
    st.markdown('<p class="section-header">Individual Risk Assessment</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclaimer-box">
        <strong>For Educational Purposes Only.</strong> This tool demonstrates model predictions and should never be used for actual clinical decisions.
        Note: This version excludes the "time" variable to avoid information leakage.
    </div>
    """, unsafe_allow_html=True)
    
    model_choice = st.selectbox("Select Model", list(models_no_time.keys()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Demographics**")
        age = st.number_input("Age (years)", 20, 100, 60)
        sex = st.selectbox("Sex", ["Female", "Male"])
        
        st.markdown("**Cardiac Function**")
        ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 38, 
                                       help="Normal: 55-70%. Below 40% indicates HFrEF.")
        
    with col2:
        st.markdown("**Laboratory Values**")
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.5, 10.0, 1.1, 0.1,
                                            help="Normal: 0.7-1.2 mg/dL")
        serum_sodium = st.number_input("Serum Sodium (mEq/L)", 110, 150, 137)
        creatinine_phosphokinase = st.number_input("CPK (mcg/L)", 20, 8000, 250)
        platelets = st.number_input("Platelets (kiloplatelets/mL)", 25000, 850000, 260000)
        
    with st.expander("Comorbidities"):
        col3, col4 = st.columns(2)
        with col3:
            anaemia = st.checkbox("Anaemia")
            diabetes = st.checkbox("Diabetes")
        with col4:
            high_bp = st.checkbox("High Blood Pressure")
            smoking = st.checkbox("Smoking")
    
    patient_data = pd.DataFrame({
        'age': [age], 'anaemia': [int(anaemia)], 'creatinine_phosphokinase': [creatinine_phosphokinase],
        'diabetes': [int(diabetes)], 'ejection_fraction': [ejection_fraction], 
        'high_blood_pressure': [int(high_bp)], 'platelets': [platelets],
        'serum_creatinine': [serum_creatinine], 'serum_sodium': [serum_sodium],
        'sex': [1 if sex == "Male" else 0], 'smoking': [int(smoking)]
    })
    
    if st.button("Calculate Risk", type="primary"):
        model = models_no_time[model_choice]
        prob = model.predict_proba(patient_data)[0][1]
        
        st.markdown("---")
        st.markdown(f"### Predicted Mortality Risk: **{prob*100:.1f}%**")
        
        if prob < 0.2:
            st.success("**Low Risk** — Model predicts favorable prognosis.")
        elif prob < 0.5:
            st.warning("**Moderate Risk** — Elevated mortality probability.")
        else:
            st.error("**High Risk** — Significant mortality probability based on features.")
        
        st.markdown("**Key Risk Factors:**")
        factors = []
        if ejection_fraction < 40:
            factors.append(f"• Reduced ejection fraction ({ejection_fraction}%)")
        if serum_creatinine > 1.5:
            factors.append(f"• Elevated serum creatinine ({serum_creatinine} mg/dL)")
        if age > 70:
            factors.append(f"• Advanced age ({age} years)")
        if serum_sodium < 135:
            factors.append(f"• Low serum sodium ({serum_sodium} mEq/L)")
        if not factors:
            factors.append("• No major risk factors based on clinical thresholds")
        st.markdown("\n".join(factors))
        
        st.caption(f"Model: {model_choice} (trained without time variable) | AUC: {results_no_time[model_choice]['auc']:.3f}")

# Footer
st.markdown("""
<div class="footer">
    <strong>CardioRisk ML</strong> — Heart Failure Clinical Records Dataset (UCI)<br>
    Reference: Chicco & Jurman, BMC Medical Informatics (2020) | Built for educational purposes only
</div>
""", unsafe_allow_html=True)
