import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WineCheck · Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

  /* ── Root theme ── */
  :root {
    --burgundy:   #6B1A2A;
    --deep-wine:  #3D0A14;
    --blush:      #F5E6D8;
    --gold:       #C8A96E;
    --cream:      #FAF6F0;
    --charcoal:   #2C2018;
    --muted:      #8A7060;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream) !important;
    color: var(--charcoal) !important;
  }

  /* ── Hide default Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, var(--deep-wine) 0%, var(--burgundy) 100%) !important;
    border-right: 2px solid var(--gold);
  }
  section[data-testid="stSidebar"] * { color: var(--blush) !important; }
  section[data-testid="stSidebar"] .stSlider > div > div > div { background: var(--gold) !important; }
  section[data-testid="stSidebar"] label { color: var(--gold) !important; font-weight: 500; font-size: 0.82rem; letter-spacing: 0.06em; text-transform: uppercase; }
  section[data-testid="stSidebar"] .stSelectbox label { color: var(--gold) !important; }

  /* ── Hero banner ── */
  .hero {
    background: linear-gradient(135deg, var(--deep-wine) 0%, var(--burgundy) 60%, #8B2940 100%);
    border-radius: 16px;
    padding: 48px 56px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(61,10,20,0.25);
  }
  .hero::before {
    content: "🍷";
    position: absolute; right: 48px; top: 50%; transform: translateY(-50%);
    font-size: 100px; opacity: 0.12;
  }
  .hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem; font-weight: 700;
    color: var(--cream); margin: 0; line-height: 1.15;
  }
  .hero p { color: var(--gold); margin: 10px 0 0; font-size: 1rem; font-weight: 300; letter-spacing: 0.04em; }

  /* ── Metric cards ── */
  .metric-grid { display: flex; gap: 16px; margin: 24px 0; flex-wrap: wrap; }
  .metric-card {
    flex: 1; min-width: 130px;
    background: white;
    border: 1px solid #EAE0D4;
    border-top: 3px solid var(--gold);
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    text-align: center;
  }
  .metric-card .val { font-family: 'Playfair Display', serif; font-size: 1.9rem; color: var(--burgundy); font-weight: 700; }
  .metric-card .lbl { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-top: 4px; }

  /* ── Prediction result ── */
  .result-high {
    background: linear-gradient(135deg, #1A4731, #2D6A4F);
    border-radius: 12px; padding: 28px 36px; margin: 20px 0;
    color: white; text-align: center;
    box-shadow: 0 6px 24px rgba(26,71,49,0.3);
  }
  .result-low {
    background: linear-gradient(135deg, var(--deep-wine), var(--burgundy));
    border-radius: 12px; padding: 28px 36px; margin: 20px 0;
    color: white; text-align: center;
    box-shadow: 0 6px 24px rgba(61,10,20,0.3);
  }
  .result-high h2, .result-low h2 { font-family: 'Playfair Display', serif; font-size: 1.9rem; margin: 0; }
  .result-high p, .result-low p { margin: 8px 0 0; opacity: 0.85; font-size: 0.95rem; }

  /* ── Section headers ── */
  .section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem; color: var(--burgundy);
    border-bottom: 2px solid var(--gold);
    padding-bottom: 8px; margin: 28px 0 18px;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: transparent; border-bottom: 2px solid #EAE0D4;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    color: var(--muted) !important; font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem; letter-spacing: 0.05em; text-transform: uppercase;
    padding: 10px 20px;
  }
  .stTabs [aria-selected="true"] {
    color: var(--burgundy) !important; font-weight: 600;
    border-bottom: 2px solid var(--burgundy) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, var(--burgundy), var(--deep-wine)) !important;
    color: var(--cream) !important; border: none !important;
    border-radius: 8px !important; padding: 12px 32px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; letter-spacing: 0.05em !important;
    font-size: 0.9rem !important; text-transform: uppercase !important;
    box-shadow: 0 4px 14px rgba(107,26,42,0.35) !important;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(107,26,42,0.45) !important;
  }

  /* ── Info box ── */
  .info-box {
    background: #FDF9F5; border: 1px solid #EAE0D4;
    border-left: 4px solid var(--gold);
    border-radius: 8px; padding: 14px 18px;
    font-size: 0.88rem; color: var(--charcoal); margin: 12px 0;
  }

  /* ── Upload area ── */
  .uploadedFile { border-color: var(--gold) !important; }

  /* ── Dataframe ── */
  .stDataFrame { border-radius: 8px; overflow: hidden; }

  /* ── Sidebar logo ── */
  .sidebar-logo {
    text-align: center; padding: 20px 0 28px;
    border-bottom: 1px solid rgba(200,169,110,0.3);
    margin-bottom: 20px;
  }
  .sidebar-logo .logo-icon { font-size: 3rem; display: block; }
  .sidebar-logo .logo-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; color: var(--blush) !important;
    letter-spacing: 0.08em;
  }
  .sidebar-logo .logo-sub { font-size: 0.72rem; color: var(--gold) !important; letter-spacing: 0.12em; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df.drop(columns=drop_cols, inplace=True)
    df["quality"] = (df["quality"] >= 6).astype(int)
    return df

@st.cache_resource
def build_pipeline(df):
    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    enc.fit(X_train[["Type"]])

    def encode_and_scale(X_part, fit_scaler=False, scaler=None):
        encoded = enc.transform(X_part[["Type"]])
        encoded_cols = enc.get_feature_names_out(["Type"])
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X_part.index)
        X_proc = pd.concat([X_part.drop(columns=["Type"]), encoded_df], axis=1)
        if fit_scaler:
            sc = StandardScaler()
            return sc.fit_transform(X_proc), sc, X_proc.columns.tolist()
        else:
            return scaler.transform(X_proc)

    X_train_sc, scaler, feature_cols = encode_and_scale(X_train, fit_scaler=True)
    X_test_sc = encode_and_scale(X_test, scaler=scaler)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42,
                                                  min_samples_split=5, min_samples_leaf=1,
                                                  max_features="log2", max_depth=None),
        "Decision Tree": DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Gaussian NB": GaussianNB()
    }

    results = {}
    fitted = {}
    for name, m in models.items():
        m.fit(X_train_sc, y_train)
        y_pred = m.predict(X_test_sc)
        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1 Score": round(f1_score(y_test, y_pred), 4),
            "y_pred": y_pred,
        }
        fitted[name] = m

    return enc, scaler, feature_cols, fitted, results, X_test_sc, y_test

def predict_single(enc, scaler, feature_cols, model, input_dict):
    row = pd.DataFrame([input_dict])
    encoded = enc.transform(row[["Type"]])
    encoded_cols = enc.get_feature_names_out(["Type"])
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols)
    row_proc = pd.concat([row.drop(columns=["Type"]), encoded_df], axis=1)
    row_sc = scaler.transform(row_proc[feature_cols])
    pred = model.predict(row_sc)[0]
    prob = model.predict_proba(row_sc)[0]
    return pred, prob

def wine_palette():
    return ["#6B1A2A", "#C8A96E", "#8B2940", "#3D0A14", "#B5875A"]

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <span class="logo-icon">🍷</span>
      <div class="logo-text">WineCheck</div>
      <div class="logo-sub">Quality Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Upload your dataset**")
    uploaded = st.file_uploader("final_alcohol.csv", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Wine Parameters**")

    wine_type = st.selectbox("Wine Type", ["White Wine", "Red Wine"])
    fixed_acid   = st.slider("Fixed Acidity",        4.0, 16.0, 7.4, 0.1)
    volatile_acid= st.slider("Volatile Acidity",     0.08, 1.6, 0.28, 0.01)
    citric       = st.slider("Citric Acid",          0.0, 1.0, 0.3, 0.01)
    residual_sug = st.slider("Residual Sugar",       0.5, 65.0, 5.0, 0.5)
    chlorides    = st.slider("Chlorides",            0.01, 0.6, 0.047, 0.001, format="%.3f")
    free_so2     = st.slider("Free Sulfur Dioxide",  1.0, 290.0, 30.0, 1.0)
    total_so2    = st.slider("Total Sulfur Dioxide", 6.0, 440.0, 115.0, 1.0)
    density      = st.slider("Density",              0.990, 1.004, 0.994, 0.0001, format="%.4f")
    ph           = st.slider("pH",                   2.7, 4.0, 3.2, 0.01)
    sulphates    = st.slider("Sulphates",            0.2, 2.0, 0.5, 0.01)
    alcohol      = st.slider("Alcohol (%)",          8.0, 15.0, 10.5, 0.1)

    user_input = {
        "fixed acidity": fixed_acid, "volatile acidity": volatile_acid,
        "citric acid": citric, "residual sugar": residual_sug,
        "chlorides": chlorides, "free sulfur dioxide": free_so2,
        "total sulfur dioxide": total_so2, "density": density,
        "pH": ph, "sulphates": sulphates, "alcohol": alcohol,
        "Type": wine_type
    }

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Wine Quality Predictor</h1>
  <p>Machine Learning · Physicochemical Analysis · Multi-Model Comparison</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class="info-box">
      📂 &nbsp; Upload <strong>final_alcohol.csv</strong> in the sidebar to unlock all features. 
      The app will train 5 ML models, display EDA, and enable real-time predictions.
    </div>
    """, unsafe_allow_html=True)

    # Feature reference table when no data uploaded
    st.markdown('<div class="section-title">Expected Dataset Features</div>', unsafe_allow_html=True)
    features_info = pd.DataFrame({
        "Feature": ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
                    "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","Type","quality"],
        "Type": ["float"]*12+["int"],
        "Description": [
            "g(tartaric acid)/dm³","g(acetic acid)/dm³","g/dm³","g/dm³","g(NaCl)/dm³",
            "mg/dm³","mg/dm³","g/cm³","log[H⁺]","g(K₂SO₄)/dm³","% vol",
            "Red Wine / White Wine","Target: 0=Low (<6), 1=High (≥6)"
        ]
    })
    st.dataframe(features_info, use_container_width=True, hide_index=True)

else:
    # ── Load & train ──────────────────────────
    df = load_data(uploaded)

    with st.spinner("Training 5 models…"):
        enc, scaler, feature_cols, fitted_models, results, X_test_sc, y_test = build_pipeline(df)

    best_model_name = max(results, key=lambda k: results[k]["F1 Score"])
    best = results[best_model_name]

    # ── KPI strip ─────────────────────────────
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="val">{len(df):,}</div>
        <div class="lbl">Samples</div>
      </div>
      <div class="metric-card">
        <div class="val">{df['quality'].mean()*100:.1f}%</div>
        <div class="lbl">High Quality</div>
      </div>
      <div class="metric-card">
        <div class="val">{best['Accuracy']*100:.1f}%</div>
        <div class="lbl">Best Accuracy</div>
      </div>
      <div class="metric-card">
        <div class="val">{best['F1 Score']:.3f}</div>
        <div class="lbl">Best F1</div>
      </div>
      <div class="metric-card">
        <div class="val">5</div>
        <div class="lbl">Models Trained</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🔮  Predict", "📊  EDA", "🤖  Model Comparison", "📋  Data"])

    # ══════════════════════════════════════════
    #  TAB 1 — PREDICT
    # ══════════════════════════════════════════
    with tab1:
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown('<div class="section-title">Choose Prediction Model</div>', unsafe_allow_html=True)
            model_choice = st.selectbox(
                "Model", list(fitted_models.keys()),
                index=list(fitted_models.keys()).index(best_model_name),
                label_visibility="collapsed"
            )
            sel_model = fitted_models[model_choice]

            st.markdown(f"""
            <div class="info-box">
              Using <strong>{model_choice}</strong> &nbsp;·&nbsp; 
              Accuracy <strong>{results[model_choice]['Accuracy']*100:.1f}%</strong> &nbsp;·&nbsp; 
              F1 <strong>{results[model_choice]['F1 Score']:.3f}</strong>
              {"&nbsp; ⭐ Best Model" if model_choice == best_model_name else ""}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Current Input Parameters**")
            input_df = pd.DataFrame([user_input]).T.rename(columns={0: "Value"})
            st.dataframe(input_df, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

            if st.button("Predict Wine Quality", use_container_width=True):
                pred, prob = predict_single(enc, scaler, feature_cols, sel_model, user_input)

                if pred == 1:
                    st.markdown(f"""
                    <div class="result-high">
                      <h2>✅ High Quality Wine</h2>
                      <p>Quality Score ≥ 6 &nbsp;·&nbsp; Confidence: <strong>{prob[1]*100:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-low">
                      <h2>⚠️ Low Quality Wine</h2>
                      <p>Quality Score &lt; 6 &nbsp;·&nbsp; Confidence: <strong>{prob[0]*100:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                # Probability gauge
                st.markdown("**Probability Breakdown**")
                fig_prob, ax_prob = plt.subplots(figsize=(5, 1.4))
                fig_prob.patch.set_facecolor('#FAF6F0')
                ax_prob.set_facecolor('#FAF6F0')
                ax_prob.barh(["Low", "High"], [prob[0], prob[1]],
                             color=["#6B1A2A", "#2D6A4F"], height=0.45, edgecolor="none")
                ax_prob.set_xlim(0, 1)
                ax_prob.set_xlabel("Probability", fontsize=9)
                for i, v in enumerate([prob[0], prob[1]]):
                    ax_prob.text(v + 0.02, i, f"{v*100:.1f}%", va="center", fontsize=9)
                ax_prob.spines[["top","right","left"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_prob, use_container_width=False)
                plt.close()
            else:
                st.markdown("""
                <div class="info-box" style="text-align:center; padding: 40px;">
                  🍾 &nbsp; Adjust the wine parameters in the sidebar, then click <strong>Predict</strong>.
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    #  TAB 2 — EDA
    # ══════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

        palette = wine_palette()

        # Row 1: Quality distribution + Wine type split
        c1, c2 = st.columns(2, gap="medium")

        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor('#FAF6F0')
            ax.set_facecolor('#FAF6F0')
            counts = df["quality"].value_counts().sort_index()
            bars = ax.bar(["Low Quality\n(score < 6)", "High Quality\n(score ≥ 6)"],
                          counts.values, color=["#6B1A2A", "#C8A96E"], edgecolor="none", width=0.5)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                        f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=10, fontweight="500")
            ax.set_title("Quality Distribution", fontsize=12, fontweight="600", pad=12)
            ax.set_ylabel("Count", fontsize=9)
            ax.spines[["top","right","left"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor('#FAF6F0')
            ax.set_facecolor('#FAF6F0')
            type_counts = df["Type"].value_counts()
            wedges, texts, autotexts = ax.pie(
                type_counts.values, labels=type_counts.index,
                colors=["#C8A96E", "#6B1A2A"],
                autopct="%1.1f%%", startangle=140,
                wedgeprops=dict(edgecolor="white", linewidth=2)
            )
            for t in autotexts: t.set_fontsize(10)
            ax.set_title("Wine Type Split", fontsize=12, fontweight="600", pad=12)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Row 2: Alcohol vs Quality
        c3, c4 = st.columns(2, gap="medium")

        num_df = df.select_dtypes(include=["int64","float64"])

        with c3:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor('#FAF6F0')
            ax.set_facecolor('#FAF6F0')
            for q, color, label in [(0, "#6B1A2A", "Low"), (1, "#C8A96E", "High")]:
                subset = df[df["quality"] == q]["alcohol"]
                ax.hist(subset, bins=20, alpha=0.75, color=color, label=label, edgecolor="white", linewidth=0.5)
            ax.set_title("Alcohol Content by Quality", fontsize=12, fontweight="600", pad=12)
            ax.set_xlabel("Alcohol (%)", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.legend(title="Quality", fontsize=9)
            ax.spines[["top","right","left"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c4:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor('#FAF6F0')
            ax.set_facecolor('#FAF6F0')
            for q, color, label in [(0, "#6B1A2A", "Low"), (1, "#C8A96E", "High")]:
                subset = df[df["quality"] == q]["volatile acidity"]
                ax.hist(subset, bins=20, alpha=0.75, color=color, label=label, edgecolor="white", linewidth=0.5)
            ax.set_title("Volatile Acidity by Quality", fontsize=12, fontweight="600", pad=12)
            ax.set_xlabel("Volatile Acidity", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.legend(title="Quality", fontsize=9)
            ax.spines[["top","right","left"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Row 3: Correlation heatmap (full width)
        st.markdown('<div class="section-title">Feature Correlation Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor('#FAF6F0')
        cmap = sns.diverging_palette(10, 140, as_cmap=True)
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap=cmap,
                    linewidths=0.5, linecolor="#FAF6F0",
                    annot_kws={"size": 8}, ax=ax,
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Pearson Correlation of Physicochemical Features", fontsize=13, fontweight="600", pad=14)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Row 4: Feature importance from RF
        st.markdown('<div class="section-title">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        rf = fitted_models["Random Forest"]
        importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#FAF6F0')
        ax.set_facecolor('#FAF6F0')
        colors = ["#C8A96E" if v >= importances.quantile(0.7) else "#6B1A2A" for v in importances.values]
        ax.barh(importances.index, importances.values, color=colors, edgecolor="none")
        ax.set_xlabel("Importance Score", fontsize=9)
        ax.set_title("Which features matter most?", fontsize=12, fontweight="600", pad=12)
        ax.spines[["top","right","bottom"]].set_visible(False)
        ax.tick_params(labelsize=9)
        gold_patch = mpatches.Patch(color='#C8A96E', label='Top 30%')
        red_patch  = mpatches.Patch(color='#6B1A2A', label='Others')
        ax.legend(handles=[gold_patch, red_patch], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════
    #  TAB 3 — MODEL COMPARISON
    # ══════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)

        # Metrics table
        metrics_df = pd.DataFrame({
            name: {k: v for k, v in res.items() if k != "y_pred"}
            for name, res in results.items()
        }).T.reset_index().rename(columns={"index": "Model"})
        metrics_df = metrics_df.sort_values("F1 Score", ascending=False).reset_index(drop=True)

        def highlight_best(s):
            is_max = s == s.max()
            return ["background-color: #FDF0D5; color: #6B1A2A; font-weight: 600" if v else "" for v in is_max]

        styled = metrics_df.style\
            .apply(highlight_best, subset=["Accuracy","Precision","Recall","F1 Score"])\
            .format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1 Score": "{:.4f}"})
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Bar chart comparison
        st.markdown('<div class="section-title">Score Breakdown</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.patch.set_facecolor('#FAF6F0')
        ax.set_facecolor('#FAF6F0')
        model_names = metrics_df["Model"].tolist()
        metric_keys = ["Accuracy", "Precision", "Recall", "F1 Score"]
        metric_colors = ["#6B1A2A", "#C8A96E", "#8B2940", "#3D0A14"]
        x = np.arange(len(model_names))
        w = 0.2
        for i, (m, c) in enumerate(zip(metric_keys, metric_colors)):
            vals = [results[n][m] for n in model_names]
            bars = ax.bar(x + i*w - 1.5*w, vals, w*0.85, label=m, color=c, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0.6, 1.0)
        ax.set_ylabel("Score", fontsize=9)
        ax.legend(fontsize=9, framealpha=0.3)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.set_title("All Models · All Metrics", fontsize=12, fontweight="600", pad=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Confusion matrix for selected model
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_model = st.selectbox("Select model for confusion matrix", list(fitted_models.keys()),
                                index=list(fitted_models.keys()).index(best_model_name))

        y_pred_cm = results[cm_model]["y_pred"]
        cm = confusion_matrix(y_test, y_pred_cm)

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#FAF6F0')
        ax.set_facecolor('#FAF6F0')
        cmap_cm = sns.light_palette("#6B1A2A", as_cmap=True)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_cm, ax=ax,
                    linewidths=2, linecolor="white",
                    xticklabels=["Low Quality", "High Quality"],
                    yticklabels=["Low Quality", "High Quality"],
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(f"{cm_model} — Confusion Matrix", fontsize=11, fontweight="600", pad=12)
        plt.tight_layout()
        col_cm, _ = st.columns([1, 1])
        with col_cm:
            st.pyplot(fig, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════
    #  TAB 4 — DATA PREVIEW
    # ══════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Rows", f"{len(df):,}")
        col_b.metric("Features", len(df.columns) - 1)
        col_c.metric("White Wine", f"{(df['Type']=='White Wine').sum():,}")

        st.dataframe(df.head(50), use_container_width=True, height=380)

        st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
        st.dataframe(df.describe().round(3), use_container_width=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:48px; padding:20px; 
            border-top:1px solid #EAE0D4; color:#8A7060; font-size:0.8rem;">
  🍷 &nbsp; WineCheck · Powered by Scikit-learn &amp; Streamlit &nbsp;·&nbsp; 
  Models: Random Forest · Decision Tree · KNN · Logistic Regression · Gaussian NB
</div>
""", unsafe_allow_html=True)