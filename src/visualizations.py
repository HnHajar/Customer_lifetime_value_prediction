"""
visualizations.py  (à placer dans src/)
Génère les graphiques CLV à partir des résultats déjà calculés par les autres modules.
"""

import os
import tempfile
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── PARTIE 1 : configuration ──────────────────────────────────────────────────
SEGMENT_COLORS = {
    "Champions":    "#2ECC71",
    "At Risk":      "#E74C3C",
    "Rising Stars": "#F39C12",
    "Dormant":      "#95A5A6",
}

# Contrat avec les autres fichiers : colonnes que df_combined / val_df doivent avoir
REQUIRED_COMBINED_COLS = ["CustomerID", "bgnbd_clv", "simple_clv_npv",
                          "prob_alive", "strategic_segment"]
REQUIRED_VAL_COLS = ["actual_future_purchases", "predicted_purchases"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 120,
})

# src/visualizations.py  ->  <projet>/outputs  (le même dossier que main.py)
OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
)

# ── PARTIE 2 : petits outils ──────────────────────────────────────────────────
def _save(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _usd(x, pos):
    return f"${x:,.0f}"


def _fmt_id(cid):
    """17850.0 -> '17850' (CustomerID est souvent un float dans Online Retail)."""
    try:
        return str(int(float(cid)))
    except (TypeError, ValueError):
        return str(cid)


def _axis_limits(series, q=0.99):
    """Bornes d'axe qui ignorent les valeurs extrêmes (distributions très asymétriques)."""
    return min(0, float(series.min())), float(series.quantile(q))

# ── PARTIE 3 : interface avec les autres fichiers ─────────────────────────────
def add_strategic_segments(df):
    """Crée 'strategic_segment' (matrice 2x2 autour des médianes) si la colonne manque."""
    df = df.copy()
    hist_high = df["simple_clv_npv"] >= df["simple_clv_npv"].median()
    fut_high = df["bgnbd_clv"] >= df["bgnbd_clv"].median()
    df["strategic_segment"] = np.select(
        [hist_high & fut_high, hist_high & ~fut_high, ~hist_high & fut_high],
        ["Champions", "At Risk", "Rising Stars"],
        default="Dormant",
    )
    return df


def build_df_combined(clv_results, df_clv_simple):
    """
    Fusionne clv_results (run_bgnbd_pipeline) et df_clv_simple (run_simple_clv_pipeline)
    sur CustomerID pour obtenir le df_combined attendu par les graphiques.
    """
    for name, d in (("clv_results", clv_results), ("df_clv_simple", df_clv_simple)):
        if "CustomerID" not in d.columns:
            raise KeyError(f"{name} n'a pas de colonne 'CustomerID' "
                           f"(colonnes : {list(d.columns)})")

    extra = [c for c in df_clv_simple.columns
             if c == "CustomerID" or c not in clv_results.columns]
    merged = clv_results.merge(df_clv_simple[extra], on="CustomerID",
                               how="inner", validate="one_to_one")
    if len(merged) < len(clv_results):
        warnings.warn(f"{len(clv_results) - len(merged)} clients de clv_results "
                      f"absents de df_clv_simple (perdus à la fusion).")

    if "strategic_segment" not in merged.columns:
        if "simple_clv_npv" not in merged.columns:
            raise KeyError("Colonne 'simple_clv_npv' introuvable après fusion : "
                           f"colonnes disponibles = {list(merged.columns)}")
        merged = add_strategic_segments(merged)
    return merged


def prepare_combined(df):
    """Vérifie le contrat de colonnes, avertit des anomalies et retire les lignes inutilisables."""
    missing = [c for c in REQUIRED_COMBINED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes dans df_combined : {missing}. "
                       f"Colonnes disponibles : {list(df.columns)}")
    if df.empty:
        raise ValueError("df_combined est vide.")

    num_cols = ["bgnbd_clv", "simple_clv_npv", "prob_alive"]
    for col in num_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"La colonne '{col}' doit être numérique (type : {df[col].dtype}).")

    if df["CustomerID"].duplicated().any():
        warnings.warn("CustomerID dupliqués dans df_combined.")

    before = len(df)
    df = df.dropna(subset=num_cols + ["strategic_segment"]).copy()
    if len(df) < before:
        warnings.warn(f"{before - len(df)} lignes avec valeurs manquantes ignorées.")
    if df.empty:
        raise ValueError("Plus aucune ligne après suppression des valeurs manquantes.")

    if not df["prob_alive"].between(0, 1).all():
        raise ValueError("prob_alive doit être comprise entre 0 et 1.")
    if (df["bgnbd_clv"] < 0).any():
        warnings.warn("CLV BG/NBD négatives : elles sont ignorées dans le camembert.")

    unknown = set(df["strategic_segment"].unique()) - set(SEGMENT_COLORS)
    if unknown:
        warnings.warn(f"Segments sans couleur définie (gris par défaut) : {unknown}")
    return df


def _resolve_metrics(val_metrics, actual, predicted):
    """Lit 'correlation' et 'mae' dans val_metrics ; sinon les recalcule (avec avertissement)."""
    corr = val_metrics.get("correlation")
    mae = val_metrics.get("mae")
    if corr is None:
        warnings.warn("Clé 'correlation' absente de val_metrics : recalculée.")
        ok = actual.std() > 0 and predicted.std() > 0
        corr = float(np.corrcoef(actual, predicted)[0, 1]) if ok else float("nan")
    if mae is None:
        warnings.warn("Clé 'mae' absente de val_metrics : recalculée.")
        mae = float(np.mean(np.abs(actual - predicted)))
    return corr, mae

# ── PARTIE 4 : graphiques 1, 2, 3 ─────────────────────────────────────────────
def plot_clv_distribution(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Customer Lifetime Value Distribution", fontsize=14, fontweight="bold")

    # Histogramme tronqué au 95e centile (sinon quelques gros clients écrasent tout)
    cap = df["bgnbd_clv"].quantile(0.95)
    ax1.hist(df.loc[df["bgnbd_clv"] <= cap, "bgnbd_clv"], bins=50, color="#3498DB",
             alpha=0.8, edgecolor="white", linewidth=0.3)
    med, mean = df["bgnbd_clv"].median(), df["bgnbd_clv"].mean()
    ax1.axvline(med, color="black", linestyle="--", linewidth=1.5, label=f"Median: ${med:,.0f}")
    ax1.axvline(mean, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"Mean: ${mean:,.0f}")
    ax1.set_title("BG/NBD CLV Distribution (95th pct cap)")
    ax1.set_xlabel("3-Year CLV ($)")
    ax1.set_ylabel("Customers")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax1.legend()

    # Part de CLV par segment (les valeurs <= 0 ne peuvent pas figurer dans un camembert)
    seg_clv = df.groupby("strategic_segment")["bgnbd_clv"].sum()
    seg_clv = seg_clv[seg_clv > 0]
    if seg_clv.empty:
        ax2.text(0.5, 0.5, "Aucune CLV positive", ha="center", va="center")
        ax2.axis("off")
    else:
        ax2.pie(seg_clv.values, labels=seg_clv.index,
                colors=[SEGMENT_COLORS.get(s, "#999") for s in seg_clv.index],
                autopct="%1.1f%%", startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax2.set_title("CLV Share by Strategic Segment")

    plt.tight_layout()
    return _save(fig, "01_clv_distribution.png")


def plot_model_comparison(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    for segment, group in df.groupby("strategic_segment"):
        ax.scatter(group["simple_clv_npv"], group["bgnbd_clv"],
                   c=SEGMENT_COLORS.get(segment, "#999"), label=segment,
                   alpha=0.5, s=15, edgecolors="none")

    x_lo, x_hi = _axis_limits(df["simple_clv_npv"], 0.95)
    y_lo, y_hi = _axis_limits(df["bgnbd_clv"], 0.95)
    lim = max(x_hi, y_hi)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="Models agree (y=x)")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("Simple CLV ($)")
    ax.set_ylabel("BG/NBD CLV ($)")
    ax.set_title("Simple CLV vs BG/NBD CLV\nPoints above line: BG/NBD is more optimistic",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    return _save(fig, "02_model_comparison.png")


def plot_probability_alive(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Probability Customer is Still Active (BG/NBD)", fontsize=14, fontweight="bold")

    ax1.hist(df["prob_alive"], bins=40, color="#9B59B6", alpha=0.8,
             edgecolor="white", linewidth=0.3)
    ax1.axvline(0.5, color="#E74C3C", linestyle="--", linewidth=1.5, label="50% threshold")
    ax1.set_xlabel("P(Customer Still Active)")
    ax1.set_ylabel("Customers")
    ax1.set_title("Distribution of P(Alive)")
    ax1.legend()

    for segment, group in df.groupby("strategic_segment"):
        ax2.hist(group["prob_alive"], bins=30, alpha=0.6,
                 color=SEGMENT_COLORS.get(segment, "#999"), label=segment)
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("P(Customer Still Active)")
    ax2.set_ylabel("Customers")
    ax2.set_title("P(Alive) by Strategic Segment")
    ax2.legend()

    plt.tight_layout()
    return _save(fig, "03_probability_alive.png")

# ── PARTIE 5 : validation du modèle (graphique 4) ─────────────────────────────
def plot_validation(val_df, val_metrics):
    missing = [c for c in REQUIRED_VAL_COLS if c not in val_df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes dans val_df : {missing}. "
                       f"Colonnes disponibles : {list(val_df.columns)}")

    actual = val_df["actual_future_purchases"].to_numpy(dtype=float)
    predicted = val_df["predicted_purchases"].to_numpy(dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[mask], predicted[mask]
    if actual.size == 0:
        raise ValueError("val_df ne contient aucune ligne exploitable (que des NaN).")
    corr, mae = _resolve_metrics(val_metrics, actual, predicted)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("BG/NBD Model Validation — Predicted vs Actual Purchases",
                 fontsize=13, fontweight="bold")

    ax1.scatter(actual, predicted, alpha=0.3, s=10, color="#3498DB", edgecolors="none")
    max_val = max(actual.max(), predicted.max())
    ax1.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
    ax1.set_xlabel("Actual Purchases (Validation Period)")
    ax1.set_ylabel("Predicted Purchases")
    ax1.set_title(f"Predicted vs Actual\nCorrelation: {corr:.3f}")
    ax1.legend()

    max_purchases = max(int(np.percentile(np.concatenate([actual, predicted]), 95)), 1)
    bins = np.arange(0, max_purchases + 2) - 0.5
    ax2.hist(actual, bins=bins, alpha=0.6, color="#E74C3C", label=f"Actual (mean={actual.mean():.2f})")
    ax2.hist(predicted, bins=bins, alpha=0.6, color="#3498DB", label=f"Predicted (mean={predicted.mean():.2f})")
    ax2.set_xlabel("Number of Purchases")
    ax2.set_ylabel("Customers")
    ax2.set_title(f"Distribution of Purchases\nMAE: {mae:.4f}")
    ax2.legend()

    plt.tight_layout()
    return _save(fig, "04_model_validation.png")

# ── PARTIE 6 : graphiques 5 et 6 ──────────────────────────────────────────────
def plot_strategic_segments(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    simple_median = df["simple_clv_npv"].median()
    bgnbd_median = df["bgnbd_clv"].median()

    for segment, group in df.groupby("strategic_segment"):
        ax.scatter(group["simple_clv_npv"], group["bgnbd_clv"],
                   c=SEGMENT_COLORS.get(segment, "#999"),
                   label=f"{segment} (n={len(group):,})",
                   alpha=0.5, s=20, edgecolors="none")

    ax.axvline(simple_median, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(bgnbd_median, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlim(*_axis_limits(df["simple_clv_npv"]))
    ax.set_ylim(*_axis_limits(df["bgnbd_clv"]))

    # Étiquettes des quadrants placées dans les coins de l'axe (transAxes = fractions 0-1)
    props = dict(fontsize=9, alpha=0.7, fontweight="bold", transform=ax.transAxes)
    ax.text(0.98, 0.97, "CHAMPIONS\nHistorical + Future ↑", ha="right", va="top",
            color=SEGMENT_COLORS["Champions"], **props)
    ax.text(0.98, 0.03, "AT RISK\nHistorical ↑, Future ↓", ha="right", va="bottom",
            color=SEGMENT_COLORS["At Risk"], **props)
    ax.text(0.02, 0.97, "RISING STARS\nHistorical ↓, Future ↑", ha="left", va="top",
            color=SEGMENT_COLORS["Rising Stars"], **props)
    ax.text(0.02, 0.03, "DORMANT\nHistorical + Future ↓", ha="left", va="bottom",
            color=SEGMENT_COLORS["Dormant"], **props)

    ax.set_xlabel("Simple CLV ($) — Historical Value")
    ax.set_ylabel("BG/NBD CLV ($) — Predicted Future Value")
    ax.set_title("Strategic CLV Segmentation Matrix", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    return _save(fig, "05_strategic_segments.png")


def plot_top_customers(df):
    top20 = df.nlargest(20, "bgnbd_clv").sort_values("bgnbd_clv")

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = [SEGMENT_COLORS.get(s, "#999") for s in top20["strategic_segment"]]
    bars = ax.barh(range(len(top20)), top20["bgnbd_clv"], color=colors, alpha=0.85)

    offset = top20["bgnbd_clv"].max() * 0.01
    for bar, val in zip(bars, top20["bgnbd_clv"]):
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}", va="center", fontsize=8)

    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f"Customer {_fmt_id(c)}" for c in top20["CustomerID"]], fontsize=8)
    ax.set_xlabel("3-Year BG/NBD CLV ($)")
    ax.set_title("Top 20 Customers by Predicted CLV", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax.legend(handles=[mpatches.Patch(color=c, label=s) for s, c in SEGMENT_COLORS.items()],
              loc="lower right", fontsize=8)

    plt.tight_layout()
    return _save(fig, "06_top_customers.png")

# ── PARTIE 7 : tableau de bord (graphique 7) ──────────────────────────────────
def plot_summary_dashboard(df, val_df, val_metrics):
    actual = val_df["actual_future_purchases"].to_numpy(dtype=float)
    predicted = val_df["predicted_purchases"].to_numpy(dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    corr, mae = _resolve_metrics(val_metrics, actual[mask], predicted[mask])

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Customer Lifetime Value — Summary Dashboard",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Haut-gauche : distribution de la CLV
    ax1 = fig.add_subplot(gs[0, 0])
    cap = df["bgnbd_clv"].quantile(0.95)
    ax1.hist(df.loc[df["bgnbd_clv"] <= cap, "bgnbd_clv"], bins=40, color="#3498DB",
             alpha=0.8, edgecolor="white", linewidth=0.3)
    ax1.axvline(df["bgnbd_clv"].median(), color="#E74C3C", linestyle="--", linewidth=1.5)
    ax1.set_title("BG/NBD CLV Distribution")
    ax1.set_xlabel("CLV ($)")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))

    # Haut-centre : matrice de segmentation
    ax2 = fig.add_subplot(gs[0, 1])
    for segment, group in df.groupby("strategic_segment"):
        ax2.scatter(group["simple_clv_npv"], group["bgnbd_clv"],
                    c=SEGMENT_COLORS.get(segment, "#999"), label=segment,
                    alpha=0.4, s=8, edgecolors="none")
    ax2.axvline(df["simple_clv_npv"].median(), color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.axhline(df["bgnbd_clv"].median(), color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_xlim(*_axis_limits(df["simple_clv_npv"]))
    ax2.set_ylim(*_axis_limits(df["bgnbd_clv"]))
    ax2.set_title("CLV Segmentation Matrix")
    ax2.set_xlabel("Simple CLV ($)")
    ax2.set_ylabel("BG/NBD CLV ($)")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax2.legend(fontsize=7)

    # Haut-droite : taille des segments
    ax3 = fig.add_subplot(gs[0, 2])
    seg_counts = df["strategic_segment"].value_counts()
    ax3.pie(seg_counts.values, labels=seg_counts.index,
            colors=[SEGMENT_COLORS.get(s, "#999") for s in seg_counts.index],
            autopct="%1.0f%%", startangle=140, textprops={"fontsize": 8},
            wedgeprops={"edgecolor": "white"})
    ax3.set_title("Segment Distribution")

    # Bas-gauche : P(alive)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(df["prob_alive"], bins=40, color="#9B59B6", alpha=0.8,
             edgecolor="white", linewidth=0.3)
    ax4.axvline(0.5, color="#E74C3C", linestyle="--", linewidth=1.5)
    ax4.set_title("P(Customer Still Active)")
    ax4.set_xlabel("Probability Alive")

    # Bas-centre : top 10 (len(top) s'adapte si moins de 10 clients)
    ax5 = fig.add_subplot(gs[1, 1])
    top = df.nlargest(10, "bgnbd_clv").sort_values("bgnbd_clv")
    ax5.barh(range(len(top)), top["bgnbd_clv"],
             color=[SEGMENT_COLORS.get(s, "#999") for s in top["strategic_segment"]], alpha=0.85)
    ax5.set_yticks(range(len(top)))
    ax5.set_yticklabels([f"C-{_fmt_id(c)[-4:]}" for c in top["CustomerID"]], fontsize=8)
    ax5.set_title("Top 10 Customers by CLV")
    ax5.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))

    # Bas-droite : indicateurs clés
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    rows = [
        ["Total Customers", f"{len(df):,}"],
        ["Avg BG/NBD CLV", f"${df['bgnbd_clv'].mean():,.2f}"],
        ["Portfolio CLV", f"${df['bgnbd_clv'].sum():,.0f}"],
        ["Avg P(Alive)", f"{df['prob_alive'].mean():.1%}"],
        ["Model MAE", f"{mae:.4f}"],
        ["Model Correlation", f"{corr:.3f}"],
    ]
    tbl = ax6.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    ax6.set_title("Key Metrics", pad=12)

    return _save(fig, "07_summary_dashboard.png")

# ── PARTIE 8 : point d'entrée (appelé depuis main.py) ─────────────────────────
def generate_all_plots(df_combined, val_df, val_metrics):
    print("\nGenerating visualizations...")
    df = prepare_combined(df_combined)
    paths = [
        plot_clv_distribution(df),
        plot_model_comparison(df),
        plot_probability_alive(df),
        plot_validation(val_df, val_metrics),
        plot_strategic_segments(df),
        plot_top_customers(df),
        plot_summary_dashboard(df, val_df, val_metrics),
    ]
    print(f"All plots saved to {OUTPUT_DIR}\n")
    return paths