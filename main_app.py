import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    page_title="MNIST Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #7C3AED; text-align: center; margin-bottom: 0.2rem; }
    .subtitle   { font-size: 1rem; color: #888; text-align: center; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    digits = load_digits()
    return digits

MODELS = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, solver="saga"),
    "Decision Tree":        DecisionTreeClassifier(max_depth=20),
    "Random Forest":        RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, max_depth=3),
    "SVM (RBF)":            SVC(probability=True, kernel="rbf", C=10, gamma=0.001),
    "SVM (Linear)":         SVC(probability=True, kernel="linear", C=0.1),
    "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":          GaussianNB(),
    "MLP Neural Network":   MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42),
}

COLORS = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A",
          "#19D3F3","#FF6692","#B6E880","#FF97FF","#FECB52"]

def compute_metrics(y_true, y_pred):
    return {
        "Accuracy":           accuracy_score(y_true, y_pred),
        "Precision (macro)":  precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (macro)":     recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (macro)":         f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (weighted)":      f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

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
    test_size    = st.slider("Tamaño del conjunto de prueba", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Semilla aleatoria", 0, 999, 42)
    scale_data   = st.checkbox("Escalar características (StandardScaler)", True)

    st.markdown("### 📊 Métricas a mostrar")
    all_metrics = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "F1 (weighted)"]
    show_metrics = st.multiselect("Métricas", all_metrics, default=["Accuracy", "F1 (macro)", "F1 (weighted)"])

    st.markdown("### 🎨 Visualizaciones")
    show_samples    = st.checkbox("Muestras del dataset", True)
    show_confusion  = st.checkbox("Matriz de Confusión", True)
    show_roc        = st.checkbox("Curvas ROC (OvR)", True)
    show_pr         = st.checkbox("Curvas Precisión-Recall", True)
    show_boundary   = st.checkbox("Frontera de Decisión (PCA 2D)", True)
    show_tsne       = st.checkbox("Proyección t-SNE", True)
    show_crossval   = st.checkbox("Validación Cruzada (K-Fold)", True)
    show_learning   = st.checkbox("Curva de Aprendizaje", True)
    show_feat_imp   = st.checkbox("Importancia de características / pesos", True)
    show_errors     = st.checkbox("Ejemplos mal clasificados", True)

    st.markdown("### 🔢 Clases ROC/PR")
    roc_classes = st.multiselect(
        "Dígitos para curvas ROC/PR",
        list(range(10)), default=[0, 1, 2, 3, 4]
    )

# ─────────────────────────────────────────────
# Load & prepare
# ─────────────────────────────────────────────
digits = load_data()
X, y   = digits.data, digits.target
class_names = [str(i) for i in range(10)]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

if scale_data:
    scaler   = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    X_all_s   = scaler.transform(X)
else:
    X_train_s, X_test_s, X_all_s = X_train, X_test, X

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🔢 MNIST Digits Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Comparación de modelos · sklearn digits dataset (8×8 px, 1797 muestras, 10 clases)</div>', unsafe_allow_html=True)

if not selected_models:
    st.warning("⚠️ Selecciona al menos un modelo en la barra lateral.")
    st.stop()

# ─────────────────────────────────────────────
# Dataset overview
# ─────────────────────────────────────────────
with st.expander("📂 Vista del Dataset", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total muestras", X.shape[0])
    c2.metric("Características (píxeles)", X.shape[1])
    c3.metric("Clases (dígitos)", 10)
    c4.metric("Resolución imagen", "8 × 8")

    if show_samples:
        st.markdown("#### Ejemplos de dígitos")
        fig_s, axes = plt.subplots(2, 10, figsize=(14, 3))
        for digit in range(10):
            idxs = np.where(y == digit)[0][:2]
            for row, idx in enumerate(idxs):
                axes[row, digit].imshow(digits.images[idx], cmap="gray_r")
                axes[row, digit].axis("off")
                if row == 0:
                    axes[row, digit].set_title(str(digit), fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_s, use_container_width=True)
        plt.close()

    # Class distribution
    dist = pd.Series(y).value_counts().sort_index()
    fig_dist = px.bar(x=dist.index.astype(str), y=dist.values,
                      labels={"x": "Dígito", "y": "Cantidad"},
                      title="Distribución de clases",
                      color=dist.values, color_continuous_scale="Purples")
    st.plotly_chart(fig_dist, use_container_width=True)

    if show_tsne:
        st.markdown("#### Proyección t-SNE del dataset completo")
        with st.spinner("Calculando t-SNE (puede tomar ~20s)..."):
            tsne  = TSNE(n_components=2, random_state=42, perplexity=30)
            X_tsne = tsne.fit_transform(X_all_s)
        df_tsne = pd.DataFrame({"x": X_tsne[:,0], "y": X_tsne[:,1], "dígito": y.astype(str)})
        fig_tsne = px.scatter(df_tsne, x="x", y="y", color="dígito",
                              title="t-SNE — MNIST (sklearn digits)",
                              color_discrete_sequence=COLORS,
                              opacity=0.7, width=800, height=500)
        st.plotly_chart(fig_tsne, use_container_width=True)

# ─────────────────────────────────────────────
# Train models
# ─────────────────────────────────────────────
results = {}
progress = st.progress(0, text="Entrenando modelos...")
for i, name in enumerate(selected_models):
    progress.progress((i) / len(selected_models), text=f"Entrenando {name}...")
    model = MODELS[name]
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s) if hasattr(model, "predict_proba") else None
    results[name] = {
        "model":   model,
        "y_pred":  y_pred,
        "y_prob":  y_prob,
        "metrics": compute_metrics(y_test, y_pred),
        "cm":      confusion_matrix(y_test, y_pred),
    }
progress.progress(1.0, text="✅ Todos los modelos entrenados")

# ─────────────────────────────────────────────
# Global metrics table + bar chart
# ─────────────────────────────────────────────
st.markdown("## 📋 Resumen de Métricas")
metric_df = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T
metric_df = metric_df[[m for m in show_metrics if m in metric_df.columns]]
if not metric_df.empty:
    st.dataframe(
        metric_df.style.background_gradient(cmap="RdYlGn", axis=0).format("{:.4f}"),
        use_container_width=True
    )
    fig_bar = px.bar(
        metric_df.reset_index().melt(id_vars="index", var_name="Métrica", value_name="Valor"),
        x="index", y="Valor", color="Métrica", barmode="group",
        labels={"index": "Modelo"}, title="Comparación de Métricas",
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
    res   = results[name]
    model = res["model"]
    y_pred = res["y_pred"]
    y_prob = res["y_prob"]
    cm     = res["cm"]

    with tab:
        # ── Quick metrics ─────────────────────────────────────
        cols = st.columns(len(res["metrics"]))
        for col, (k, v) in zip(cols, res["metrics"].items()):
            col.metric(k, f"{v:.4f}")

        # ── Confusion Matrix ──────────────────────────────────
        if show_confusion:
            st.markdown("### Matriz de Confusión")
            fig_cm, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names,
                        linewidths=0.3, ax=ax)
            ax.set_xlabel("Predicho", fontsize=12)
            ax.set_ylabel("Real", fontsize=12)
            ax.set_title(f"Confusion Matrix — {name}", fontsize=13)
            st.pyplot(fig_cm)
            plt.close()

            # Normalized confusion matrix
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fig_cmn, ax2 = plt.subplots(figsize=(9, 7))
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                        xticklabels=class_names, yticklabels=class_names,
                        linewidths=0.3, ax=ax2, vmin=0, vmax=1)
            ax2.set_xlabel("Predicho"); ax2.set_ylabel("Real")
            ax2.set_title(f"Confusion Matrix Normalizada — {name}")
            st.pyplot(fig_cmn)
            plt.close()

            with st.expander("📝 Classification Report"):
                st.text(classification_report(y_test, y_pred, target_names=class_names))

        # ── ROC Curves ───────────────────────────────────────
        if show_roc and y_prob is not None and roc_classes:
            st.markdown("### Curvas ROC (One-vs-Rest)")
            y_bin = label_binarize(y_test, classes=list(range(10)))
            fig_roc = go.Figure()
            for i in roc_classes:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"Dígito {i} (AUC={roc_auc:.3f})",
                    line=dict(width=2, color=COLORS[i])
                ))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                name="Random", line=dict(dash="dash", color="gray", width=1)))
            fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR",
                title=f"ROC Curves — {name}", height=420)
            st.plotly_chart(fig_roc, use_container_width=True)

        # ── Precision-Recall ──────────────────────────────────
        if show_pr and y_prob is not None and roc_classes:
            st.markdown("### Curvas Precisión-Recall")
            y_bin = label_binarize(y_test, classes=list(range(10)))
            fig_pr = go.Figure()
            for i in roc_classes:
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                ap = average_precision_score(y_bin[:, i], y_prob[:, i])
                fig_pr.add_trace(go.Scatter(
                    x=rec, y=prec, mode="lines", name=f"Dígito {i} (AP={ap:.3f})",
                    line=dict(width=2, color=COLORS[i])
                ))
            fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                title=f"Precision-Recall — {name}", height=420)
            st.plotly_chart(fig_pr, use_container_width=True)

        # ── Decision Boundary PCA ─────────────────────────────
        if show_boundary:
            st.markdown("### Frontera de Decisión (PCA 2D)")
            pca = PCA(n_components=2, random_state=42)
            X_tr2 = pca.fit_transform(X_train_s)
            X_te2 = pca.transform(X_test_s)
            X_all2 = pca.transform(X_all_s)

            m2d = MODELS[name].__class__(**MODELS[name].get_params())
            m2d.fit(X_tr2, y_train)

            h = 0.4
            x_min, x_max = X_all2[:,0].min()-1, X_all2[:,0].max()+1
            y_min, y_max = X_all2[:,1].min()-1, X_all2[:,1].max()+1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                  np.arange(y_min, y_max, h))
            Z = m2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            fig_bd, ax = plt.subplots(figsize=(8, 6))
            ax.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.tab10, levels=np.arange(-0.5, 10, 1))
            sc_cmap = plt.cm.tab10
            for digit in range(10):
                mask = y == digit
                ax.scatter(X_all2[mask, 0], X_all2[mask, 1],
                           c=[sc_cmap(digit/10)], label=str(digit),
                           s=15, alpha=0.6, edgecolors="none")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax.set_title(f"Frontera de Decisión PCA — {name}")
            ax.legend(title="Dígito", bbox_to_anchor=(1.01, 1), loc="upper left",
                      markerscale=2, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_bd)
            plt.close()

        # ── Cross Validation ──────────────────────────────────
        if show_crossval:
            st.markdown("### Validación Cruzada (5-Fold Stratified)")
            with st.spinner("Calculando CV..."):
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(model, X_all_s, y, cv=cv, scoring="accuracy")
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(
                x=[f"Fold {i+1}" for i in range(5)], y=cv_scores,
                marker_color=COLORS[:5],
                text=[f"{s:.4f}" for s in cv_scores], textposition="outside"
            ))
            fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash",
                             annotation_text=f"Media={cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
                             annotation_position="top right")
            fig_cv.update_layout(
                title=f"CV Accuracy — {name}", yaxis_range=[0, 1.05], height=350
            )
            st.plotly_chart(fig_cv, use_container_width=True)

        # ── Learning Curve ────────────────────────────────────
        if show_learning:
            st.markdown("### Curva de Aprendizaje")
            with st.spinner("Calculando curva de aprendizaje..."):
                train_sizes, train_scores, val_scores = learning_curve(
                    MODELS[name].__class__(**MODELS[name].get_params()),
                    X_all_s, y,
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
                    n_jobs=-1, scoring="accuracy",
                    train_sizes=np.linspace(0.1, 1.0, 8)
                )
            tr_mean = train_scores.mean(axis=1)
            tr_std  = train_scores.std(axis=1)
            val_mean = val_scores.mean(axis=1)
            val_std  = val_scores.std(axis=1)

            fig_lc = go.Figure([
                go.Scatter(x=np.concatenate([train_sizes, train_sizes[::-1]]),
                           y=np.concatenate([tr_mean+tr_std, (tr_mean-tr_std)[::-1]]),
                           fill="toself", fillcolor="rgba(99,110,250,0.15)",
                           line=dict(color="rgba(255,255,255,0)"), showlegend=False),
                go.Scatter(x=train_sizes, y=tr_mean, mode="lines+markers",
                           name="Train", line=dict(color=COLORS[0], width=2)),
                go.Scatter(x=np.concatenate([train_sizes, train_sizes[::-1]]),
                           y=np.concatenate([val_mean+val_std, (val_mean-val_std)[::-1]]),
                           fill="toself", fillcolor="rgba(239,85,59,0.15)",
                           line=dict(color="rgba(255,255,255,0)"), showlegend=False),
                go.Scatter(x=train_sizes, y=val_mean, mode="lines+markers",
                           name="Validación", line=dict(color=COLORS[1], width=2)),
            ])
            fig_lc.update_layout(
                xaxis_title="Tamaño del conjunto de entrenamiento",
                yaxis_title="Accuracy", title=f"Curva de Aprendizaje — {name}",
                yaxis_range=[0, 1.05], height=380
            )
            st.plotly_chart(fig_lc, use_container_width=True)

        # ── Feature Importance / Weights ──────────────────────
        if show_feat_imp:
            feat_imp = None
            label_fi = "Importancia"
            if hasattr(model, "feature_importances_"):
                feat_imp = model.feature_importances_
            elif hasattr(model, "coef_"):
                feat_imp = np.abs(model.coef_).mean(axis=0)
                label_fi = "|Coef| promedio"

            if feat_imp is not None:
                st.markdown(f"### {label_fi} de Píxeles (8×8)")
                img = feat_imp.reshape(8, 8)
                fig_fi, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(img, cmap="hot", interpolation="nearest")
                plt.colorbar(im, ax=ax)
                ax.set_title(f"{label_fi} — {name}")
                ax.axis("off")
                st.pyplot(fig_fi)
                plt.close()

        # ── Misclassified examples ────────────────────────────
        if show_errors:
            st.markdown("### Ejemplos Mal Clasificados")
            wrong_idx = np.where(y_pred != y_test)[0]
            n_show = min(20, len(wrong_idx))
            if n_show == 0:
                st.success("✅ ¡Sin errores en el conjunto de prueba!")
            else:
                fig_err, axes = plt.subplots(2, 10, figsize=(14, 3.5))
                axes = axes.flatten()
                for k in range(min(20, len(wrong_idx))):
                    idx = wrong_idx[k]
                    img_vec = X_test[idx]
                    axes[k].imshow(img_vec.reshape(8, 8), cmap="gray_r")
                    axes[k].set_title(f"V:{y_test[idx]} P:{y_pred[idx]}", fontsize=8, color="red")
                    axes[k].axis("off")
                for k in range(min(20, len(wrong_idx)), 20):
                    axes[k].axis("off")
                plt.suptitle(f"Errores ({len(wrong_idx)} de {len(y_test)} muestras de prueba)", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig_err)
                plt.close()

# ─────────────────────────────────────────────
# Global Radar chart
# ─────────────────────────────────────────────
st.markdown("## 🕸️ Comparación Global — Radar Chart")
radar_metrics = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "F1 (weighted)"]
fig_radar = go.Figure()
for i, (name, res) in enumerate(results.items()):
    vals = [res["metrics"].get(m, 0) for m in radar_metrics]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=radar_metrics + [radar_metrics[0]],
        fill="toself", name=name, line=dict(color=COLORS[i % len(COLORS)])
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="Comparación Radar de Métricas", height=520
)
st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────
# Heatmap accuracy per digit per model
# ─────────────────────────────────────────────
st.markdown("## 🎯 Accuracy por Dígito y Modelo")
per_digit = {}
for name, res in results.items():
    row = {}
    for digit in range(10):
        mask = y_test == digit
        row[str(digit)] = accuracy_score(y_test[mask], res["y_pred"][mask]) if mask.sum() > 0 else 0
    per_digit[name] = row
df_pd = pd.DataFrame(per_digit).T
fig_hm = px.imshow(df_pd, text_auto=".2f", aspect="auto",
                   color_continuous_scale="RdYlGn", zmin=0, zmax=1,
                   title="Accuracy por clase (dígito) y modelo",
                   labels=dict(x="Dígito", y="Modelo", color="Accuracy"))
fig_hm.update_layout(height=max(250, len(selected_models)*60))
st.plotly_chart(fig_hm, use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#666;'>🔢 MNIST Digits Dashboard · "
    "Construido con Streamlit + scikit-learn</p>",
    unsafe_allow_html=True
)
