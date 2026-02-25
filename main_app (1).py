import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #4A90D9; text-align: center; margin-bottom: 0.2rem; }
    .subtitle  { font-size: 1rem; color: #888; text-align: center; margin-bottom: 1.5rem; }
    .metric-card { background: #1e2130; border-radius: 12px; padding: 1rem; text-align: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data & helpers
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = df["target"].map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})
    return df, iris

MODELS = {
    "Logistic Regression":    LogisticRegression(max_iter=500),
    "Decision Tree":          DecisionTreeClassifier(),
    "Random Forest":          RandomForestClassifier(n_estimators=100),
    "Gradient Boosting":      GradientBoostingClassifier(),
    "SVM (RBF)":              SVC(probability=True, kernel="rbf"),
    "SVM (Linear)":           SVC(probability=True, kernel="linear"),
    "K-Nearest Neighbors":    KNeighborsClassifier(),
    "Naive Bayes":            GaussianNB(),
    "MLP Neural Network":     MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500),
}

COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF"]

def get_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "Accuracy":          accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (macro)":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (macro)":        f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return metrics

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    st.markdown("### 🤖 Modelos")
    selected_models = st.multiselect(
        "Selecciona modelos",
        list(MODELS.keys()),
        default=["Logistic Regression", "Random Forest", "SVM (RBF)"]
    )

    st.markdown("### 🧪 División de datos")
    test_size = st.slider("Tamaño del conjunto de prueba", 0.1, 0.5, 0.25, 0.05)
    random_state = st.number_input("Semilla aleatoria", 0, 999, 42)
    scale_data = st.checkbox("Escalar características (StandardScaler)", True)

    st.markdown("### 📊 Métricas de desempeño")
    show_metrics = st.multiselect(
        "Mostrar métricas",
        ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"],
        default=["Accuracy", "F1 (macro)"]
    )

    st.markdown("### 🎨 Visualizaciones")
    show_confusion  = st.checkbox("Matriz de Confusión", True)
    show_roc        = st.checkbox("Curvas ROC", True)
    show_pr         = st.checkbox("Curvas Precisión-Recall", True)
    show_boundary   = st.checkbox("Frontera de Decisión (PCA 2D)", True)
    show_boundary_feat = st.checkbox("Frontera por características seleccionadas", True)
    show_crossval   = st.checkbox("Validación Cruzada (K-Fold)", True)
    show_feature_imp = st.checkbox("Importancia de características", True)
    show_scatter    = st.checkbox("Scatter Plot del dataset", True)

    st.markdown("### 🔄 Frontera de decisión")
    feat_x = st.selectbox("Feature X", ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"], index=2)
    feat_y = st.selectbox("Feature Y", ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"], index=3)

# ─────────────────────────────────────────────
# Load & prepare data
# ─────────────────────────────────────────────
df, iris = load_data()
X = iris.data
y = iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

if scale_data:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    X_all_s   = scaler.transform(X)
else:
    X_train_s, X_test_s, X_all_s = X_train, X_test, X

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🌸 Iris Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Comparación de modelos de clasificación con múltiples métricas y visualizaciones</div>', unsafe_allow_html=True)

if not selected_models:
    st.warning("⚠️ Selecciona al menos un modelo en la barra lateral.")
    st.stop()

# ─────────────────────────────────────────────
# Train models
# ─────────────────────────────────────────────
results = {}
with st.spinner("Entrenando modelos..."):
    for name in selected_models:
        model = MODELS[name]
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s) if hasattr(model, "predict_proba") else None
        results[name] = {
            "model":  model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": get_metrics(y_test, y_pred, y_prob),
            "cm":     confusion_matrix(y_test, y_pred),
        }

# ─────────────────────────────────────────────
# Dataset overview
# ─────────────────────────────────────────────
with st.expander("📂 Vista del Dataset", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Total muestras", len(df))
    c2.metric("Características", len(iris.feature_names))
    c3.metric("Clases", len(class_names))
    st.dataframe(df.describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)

    if show_scatter:
        fig_s = px.scatter_matrix(
            df, dimensions=iris.feature_names, color="species",
            color_discrete_sequence=COLORS, title="Scatter Matrix - Iris Dataset"
        )
        fig_s.update_traces(diagonal_visible=False, marker=dict(size=4, opacity=0.7))
        st.plotly_chart(fig_s, use_container_width=True)

# ─────────────────────────────────────────────
# Metrics summary table
# ─────────────────────────────────────────────
st.markdown("## 📋 Resumen de Métricas")
metric_df = pd.DataFrame({name: res["metrics"] for name, res in results.items()}).T
metric_df = metric_df[[m for m in show_metrics if m in metric_df.columns]]
if not metric_df.empty:
    st.dataframe(
        metric_df.style.background_gradient(cmap="RdYlGn", axis=0).format("{:.4f}"),
        use_container_width=True
    )

    # Bar chart comparison
    fig_bar = px.bar(
        metric_df.reset_index().melt(id_vars="index", var_name="Métrica", value_name="Valor"),
        x="index", y="Valor", color="Métrica", barmode="group",
        labels={"index": "Modelo"}, title="Comparación de Métricas por Modelo",
        color_discrete_sequence=COLORS
    )
    fig_bar.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────
# Per-model tabs
# ─────────────────────────────────────────────
st.markdown("## 🔍 Análisis por Modelo")
tabs = st.tabs(selected_models)

for tab, name in zip(tabs, selected_models):
    res = results[name]
    model   = res["model"]
    y_pred  = res["y_pred"]
    y_prob  = res["y_prob"]
    cm      = res["cm"]

    with tab:
        # Quick metrics
        cols = st.columns(4)
        metric_vals = res["metrics"]
        for i, (k, v) in enumerate(metric_vals.items()):
            cols[i % 4].metric(k, f"{v:.4f}")

        # ── Confusion Matrix ──────────────────────────────────
        if show_confusion:
            st.markdown("### Matriz de Confusión")
            fig_cm, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
            ax.set_title(f"Confusion Matrix — {name}")
            st.pyplot(fig_cm, use_container_width=False)
            plt.close()

            st.text(classification_report(y_test, y_pred, target_names=class_names))

        # ── ROC Curves ───────────────────────────────────────
        if show_roc and y_prob is not None:
            st.markdown("### Curvas ROC (One-vs-Rest)")
            y_bin = label_binarize(y_test, classes=[0, 1, 2])
            fig_roc = go.Figure()
            for i, cls in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                    name=f"{cls} (AUC={roc_auc:.3f})", line=dict(width=2)))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                name="Random", line=dict(dash="dash", color="gray")))
            fig_roc.update_layout(
                xaxis_title="FPR", yaxis_title="TPR",
                title=f"Curvas ROC — {name}", height=400
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # ── Precision-Recall ──────────────────────────────────
        if show_pr and y_prob is not None:
            st.markdown("### Curvas Precisión-Recall")
            y_bin = label_binarize(y_test, classes=[0, 1, 2])
            fig_pr = go.Figure()
            for i, cls in enumerate(class_names):
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                ap = average_precision_score(y_bin[:, i], y_prob[:, i])
                fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                    name=f"{cls} (AP={ap:.3f})", line=dict(width=2)))
            fig_pr.update_layout(
                xaxis_title="Recall", yaxis_title="Precision",
                title=f"Precision-Recall — {name}", height=400
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        # ── Decision Boundary (PCA 2D) ────────────────────────
        if show_boundary:
            st.markdown("### Frontera de Decisión (PCA 2D)")
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_all_s)
            X_tr_2d = pca.transform(X_train_s)
            X_te_2d = pca.transform(X_test_s)

            # Retrain on PCA
            m2d = MODELS[name].__class__(**MODELS[name].get_params())
            m2d.fit(X_tr_2d, y_train)

            h = 0.05
            x_min, x_max = X_2d[:,0].min()-0.5, X_2d[:,0].max()+0.5
            y_min, y_max = X_2d[:,1].min()-0.5, X_2d[:,1].max()+0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = m2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            fig_bd, ax2 = plt.subplots(figsize=(7, 5))
            cmap_bg = plt.cm.RdYlBu
            ax2.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg)
            scatter_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            for i, cls in enumerate(class_names):
                mask = y == i
                ax2.scatter(X_2d[mask,0], X_2d[mask,1], c=scatter_colors[i],
                            label=cls, edgecolors="k", linewidths=0.4, s=50)
            ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax2.set_title(f"Frontera de Decisión PCA — {name}")
            ax2.legend()
            st.pyplot(fig_bd, use_container_width=False)
            plt.close()

        # ── Decision Boundary (feature pair) ─────────────────
        if show_boundary_feat:
            st.markdown(f"### Frontera de Decisión: {feat_x} vs {feat_y}")
            feat_names = list(iris.feature_names)
            ix = feat_names.index(feat_x)
            iy = feat_names.index(feat_y)

            X2 = X[:, [ix, iy]]
            if scale_data:
                sc2 = StandardScaler()
                X2_tr = sc2.fit_transform(X_train[:, [ix, iy]])
                X2_te = sc2.transform(X_test[:, [ix, iy]])
                X2_all = sc2.transform(X2)
            else:
                X2_tr, X2_te, X2_all = X_train[:,[ix,iy]], X_test[:,[ix,iy]], X2

            mf = MODELS[name].__class__(**MODELS[name].get_params())
            mf.fit(X2_tr, y_train)

            h = 0.05
            x_min, x_max = X2_all[:,0].min()-0.5, X2_all[:,0].max()+0.5
            y_min2, y_max2 = X2_all[:,1].min()-0.5, X2_all[:,1].max()+0.5
            xx2, yy2 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min2, y_max2, h))
            Z2 = mf.predict(np.c_[xx2.ravel(), yy2.ravel()]).reshape(xx2.shape)

            fig_bf, ax3 = plt.subplots(figsize=(7, 5))
            ax3.contourf(xx2, yy2, Z2, alpha=0.3, cmap=plt.cm.RdYlBu)
            for i, cls in enumerate(class_names):
                mask = y == i
                ax3.scatter(X2_all[mask,0], X2_all[mask,1],
                            c=scatter_colors[i], label=cls,
                            edgecolors="k", linewidths=0.4, s=50)
            ax3.set_xlabel(feat_x); ax3.set_ylabel(feat_y)
            ax3.set_title(f"Frontera de Decisión — {name}")
            ax3.legend()
            st.pyplot(fig_bf, use_container_width=False)
            plt.close()

        # ── Cross Validation ──────────────────────────────────
        if show_crossval:
            st.markdown("### Validación Cruzada (5-Fold Stratified)")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(model, X_all_s, y, cv=cv, scoring="accuracy")
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(
                x=[f"Fold {i+1}" for i in range(5)],
                y=cv_scores, marker_color=COLORS[:5],
                text=[f"{s:.3f}" for s in cv_scores], textposition="outside"
            ))
            fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash",
                             annotation_text=f"Media={cv_scores.mean():.4f}",
                             annotation_position="top right")
            fig_cv.update_layout(
                title=f"CV Scores — {name}  |  Mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
                yaxis_range=[0, 1.05], height=350
            )
            st.plotly_chart(fig_cv, use_container_width=True)

        # ── Feature Importance ────────────────────────────────
        if show_feature_imp:
            feat_imp = None
            if hasattr(model, "feature_importances_"):
                feat_imp = model.feature_importances_
            elif hasattr(model, "coef_"):
                feat_imp = np.abs(model.coef_).mean(axis=0)

            if feat_imp is not None:
                st.markdown("### Importancia / Coeficientes de Características")
                fig_fi = px.bar(
                    x=feat_imp, y=iris.feature_names,
                    orientation="h", color=feat_imp,
                    color_continuous_scale="Blues",
                    labels={"x": "Importancia", "y": "Característica"},
                    title=f"Feature Importance — {name}"
                )
                fig_fi.update_layout(height=300)
                st.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────────────────────
# Global comparison: Radar chart
# ─────────────────────────────────────────────
st.markdown("## 🕸️ Comparación Global — Radar Chart")
metrics_all = {m: res["metrics"] for m, res in results.items()}
radar_metrics = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
fig_radar = go.Figure()
for i, (name, mvals) in enumerate(metrics_all.items()):
    values = [mvals[m] for m in radar_metrics] + [mvals[radar_metrics[0]]]
    fig_radar.add_trace(go.Scatterpolar(
        r=values, theta=radar_metrics + [radar_metrics[0]],
        fill="toself", name=name, line=dict(color=COLORS[i % len(COLORS)])
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="Comparación Radar de Métricas", height=500
)
st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#666;'>🌸 Iris Classification Dashboard · "
    "Construido con Streamlit + scikit-learn</p>",
    unsafe_allow_html=True
)
