
##############################################################
# IMPORTS
##############################################################

import numpy as np
import pandas as pd

from scipy.special import hyp2f1
from lifetimes import BetaGeoFitter, GammaGammaFitter


##############################################################
# CONSTANTES
##############################################################

GROSS_MARGIN = 0.40
DISCOUNT_RATE = 0.10
PREDICTION_DAYS = 90
PREDICTION_YEARS = 3


##############################################################
# 1. ENTRAINEMENT BG/NBD
##############################################################

def fit_bgnbd(summary: pd.DataFrame) -> BetaGeoFitter:
    """
    Entraine BG/NBD pour predire les achats futurs.
    """

    print("\nFitting BG/NBD model...")

    bgf = BetaGeoFitter(penalizer_coef=0.01)

    bgf.fit(
        summary["frequency"],
        summary["recency"],
        summary["T"],
        verbose=False
    )

    print("BG/NBD fitted successfully")

    print(
        f"Parameters: "
        f"r={bgf.params_['r']:.4f}, "
        f"alpha={bgf.params_['alpha']:.4f}, "
        f"a={bgf.params_['a']:.4f}, "
        f"b={bgf.params_['b']:.4f}"
    )

    # a et b decrivent la probabilite d'abandon.
    # Probabilite moyenne d'abandon = a / (a + b).

    return bgf


##############################################################
# 2. FORMULE BG/NBD DE SECOURS
##############################################################

def _predict_bgnbd_formula(bgf, days, frequency, recency, T):
    """
    Formule analytique BG/NBD pour les achats futurs.

    Utilisee si Lifetimes retourne des valeurs NaN.
    Evite le logarithme d'un terme negatif qui peut
    poser probleme dans certaines implementations.
    """

    r = float(bgf.params_["r"])
    alpha = float(bgf.params_["alpha"])
    a = float(bgf.params_["a"])
    b = float(bgf.params_["b"])

    x = np.asarray(frequency, dtype=float)
    tx = np.asarray(recency, dtype=float)
    age = np.asarray(T, dtype=float)

    z = days / (alpha + age + days)

    # Partie principale de l'esperance BG/NBD
    coefficient = (a + b + x - 1) / (a - 1)

    hypergeom = hyp2f1(
        r + x,
        b + x,
        a + b + x - 1,
        z
    )

    term = 1 - (
        (alpha + age) / (alpha + age + days)
    ) ** (r + x) * hypergeom

    # Denominateur conditionnel
    denominator = np.ones_like(x, dtype=float)

    repeat_mask = x > 0

    denominator[repeat_mask] = 1 + (
        a / (b + x[repeat_mask] - 1)
    ) * (
        (alpha + age[repeat_mask])
        / (alpha + tx[repeat_mask])
    ) ** (r + x[repeat_mask])

    return coefficient * term / denominator


##############################################################
# 3. PREDIRE LES ACHATS FUTURS
##############################################################

def predict_purchases(
    bgf: BetaGeoFitter,
    summary: pd.DataFrame,
    days: int = PREDICTION_DAYS
) -> pd.DataFrame:
    """
    Ajoute :
    - predicted_purchases
    - prob_alive
    """

    df = summary.copy()

    if days <= 0:
        raise ValueError("days doit etre positif")

    # Prediction standard Lifetimes
    df["predicted_purchases"] = (
        bgf.conditional_expected_number_of_purchases_up_to_time(
            days,
            df["frequency"],
            df["recency"],
            df["T"]
        )
    )

    # Detection des valeurs invalides
    invalid_mask = ~np.isfinite(
        df["predicted_purchases"].to_numpy(dtype=float)
    )

    if invalid_mask.any():

        print(
            f"\nCorrection de {invalid_mask.sum()} "
            "predictions BG/NBD invalides..."
        )

        subset = df.loc[invalid_mask]

        corrected = _predict_bgnbd_formula(
            bgf,
            days,
            subset["frequency"],
            subset["recency"],
            subset["T"]
        )

        df.loc[invalid_mask, "predicted_purchases"] = corrected

    # Verification finale
    values = df["predicted_purchases"].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "BG/NBD contient encore des predictions invalides"
        )

    if (values < 0).any():
        raise ValueError(
            "BG/NBD contient des predictions negatives"
        )

    # Probabilite d'activite
    df["prob_alive"] = bgf.conditional_probability_alive(
        df["frequency"],
        df["recency"],
        df["T"]
    )

    alive = df["prob_alive"].to_numpy(dtype=float)

    if not np.isfinite(alive).all():
        raise ValueError("Probabilites d'activite invalides")

    if not df["prob_alive"].between(0, 1).all():
        raise ValueError("Probabilites hors de [0, 1]")

    print(f"\nPurchase predictions ({days}-day horizon):")

    print(
        f"Avg predicted purchases: "
        f"{df['predicted_purchases'].mean():.3f}"
    )

    print(
        f"Avg probability alive: "
        f"{df['prob_alive'].mean():.3f}"
    )

    print(
        f"Customers likely alive: "
        f"{(df['prob_alive'] > 0.5).sum():,} "
        f"({(df['prob_alive'] > 0.5).mean():.1%})"
    )

    return df


##############################################################
# 4. ENTRAINEMENT GAMMA-GAMMA
##############################################################



def calculate_repeat_monetary(
    df_train: pd.DataFrame,
    summary: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcule le montant moyen des achats repetes
    en excluant la premiere facture de chaque client.
    """

    orders = (
        df_train.groupby(
            ["CustomerID", "InvoiceNo"],
            as_index=False
        )
        .agg(
            InvoiceDate=("InvoiceDate", "min"),
            OrderRevenue=("OrderRevenue", "sum")
        )
        .sort_values(
            ["CustomerID", "InvoiceDate", "InvoiceNo"]
        )
    )

    orders["order_rank"] = (
        orders.groupby("CustomerID").cumcount()
    )

    repeat_orders = orders[
        orders["order_rank"] > 0
    ]

    monetary = (
        repeat_orders.groupby("CustomerID")["OrderRevenue"]
        .mean()
        .reset_index(name="repeat_monetary")
    )

    result = summary.copy()

    result["CustomerID"] = result["CustomerID"].astype(str)
    monetary["CustomerID"] = monetary["CustomerID"].astype(str)

    result = result.merge(
        monetary,
        on="CustomerID",
        how="left",
        validate="one_to_one"
    )

    repeat_mask = result["frequency"] > 0

    if result.loc[repeat_mask, "repeat_monetary"].isna().any():
        raise ValueError(
            "Montants des achats repetes manquants"
        )

    print(
        "\nMontant moyen des achats repetes :",
        round(
            result.loc[repeat_mask, "repeat_monetary"].mean(),
            2
        )
    )

    return result




def fit_gamma_gamma(
    summary: pd.DataFrame
) -> GammaGammaFitter:
    """
    Entraine Gamma-Gamma sur les clients repetitifs.
    """

    print("\nFitting Gamma-Gamma model...")

    repeat_buyers = summary[
    (summary["frequency"] > 0)
    & (summary["repeat_monetary"] > 0)
].copy()

    print(
        f"Using {len(repeat_buyers):,} "
        "repeat buyers for fitting"
    )

    if repeat_buyers.empty:
        raise ValueError(
            "Aucun client valide pour Gamma-Gamma"
        )

    ggf = GammaGammaFitter(penalizer_coef=0.01)

    ggf.fit(
    repeat_buyers["frequency"],
    repeat_buyers["repeat_monetary"],
    verbose=False
)

    print("Gamma-Gamma fitted successfully")

    print(
        f"Parameters: "
        f"p={ggf.params_['p']:.4f}, "
        f"q={ggf.params_['q']:.4f}, "
        f"v={ggf.params_['v']:.4f}"
    )

    return ggf


##############################################################
# 5. CALCULER LA CLV
##############################################################

def predict_clv(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    summary: pd.DataFrame
) -> pd.DataFrame:
    """
    CLV = achats futurs x panier attendu x marge.

    Les achats sont actualises mois par mois.
    """

    print("\nCalculating Customer Lifetime Value...")

    df = summary.copy()

    repeat_mask = (
    (df["frequency"] > 0)
    & (df["repeat_monetary"] > 0)
)

    if not repeat_mask.any():
        raise ValueError("Aucun client repetitif valide")

    # Panier moyen attendu pour les clients repetitifs
    
    expected_values = ggf.conditional_expected_average_profit(
    df.loc[repeat_mask, "frequency"],
    df.loc[repeat_mask, "repeat_monetary"]
)


    df["predicted_avg_value"] = np.nan

    df.loc[repeat_mask, "predicted_avg_value"] = (
        np.asarray(expected_values, dtype=float)
    )

    # Pour les clients sans achat repete :
    # estimation avec la moyenne des paniers predits
    # des clients repetitifs.
    population_avg = df.loc[
        repeat_mask, "predicted_avg_value"
    ].mean()

    if not np.isfinite(population_avg) or population_avg <= 0:
        raise ValueError("Panier moyen estime invalide")

    df["predicted_avg_value"] = (
        df["predicted_avg_value"].fillna(population_avg)
    )

    # CLV sur 3 ans, calculee mois par mois
    months = PREDICTION_YEARS * 12

    total_clv = np.zeros(len(df), dtype=float)
    previous_purchases = np.zeros(len(df), dtype=float)

    for month in range(1, months + 1):

        # Approximation : 365 jours par an
        horizon_days = month * 365 / 12

        cumulative = _predict_bgnbd_formula(
            bgf,
            horizon_days,
            df["frequency"],
            df["recency"],
            df["T"]
        )

        # Achats attendus pendant ce mois
        monthly_purchases = cumulative - previous_purchases

        if not np.isfinite(monthly_purchases).all():
            raise ValueError(
                f"Predictions invalides au mois {month}"
            )

        if (monthly_purchases < -1e-8).any():
            raise ValueError(
                f"Predictions decroissantes au mois {month}"
            )

        monthly_purchases = np.maximum(monthly_purchases, 0)

        # Actualisation mensuelle
        discount_factor = (
            1 + DISCOUNT_RATE / 12
        ) ** month

        total_clv += (
            monthly_purchases
            * df["predicted_avg_value"].to_numpy(dtype=float)
            * GROSS_MARGIN
            / discount_factor
        )

        previous_purchases = cumulative

    df["bgnbd_clv"] = total_clv

    if not np.isfinite(df["bgnbd_clv"]).all():
        raise ValueError("CLV contient des valeurs invalides")

    # Segmentation en quatre groupes
    # rank evite les problemes de valeurs identiques
    df["clv_segment"] = pd.qcut(
        df["bgnbd_clv"].rank(method="first"),
        q=4,
        labels=[
            "Low Value",
            "Mid Value",
            "High Value",
            "Top Value"
        ]
    )

    print(
        f"\nBG/NBD CLV Results "
        f"({PREDICTION_YEARS}-year horizon):"
    )

    print(
        f"Avg CLV per customer: "
        f"{df['bgnbd_clv'].mean():,.2f}"
    )

    print(
        f"Total portfolio CLV: "
        f"{df['bgnbd_clv'].sum():,.2f}"
    )

    print(
        f"Median CLV: "
        f"{df['bgnbd_clv'].median():,.2f}"
    )

    return df


##############################################################
# 6. VALIDATION BG/NBD
##############################################################

def validate_model(
    bgf: BetaGeoFitter,
    summary_valid: pd.DataFrame,
    days: int = PREDICTION_DAYS
) -> tuple:
    """
    Compare achats predits et achats reels.
    Calcule MAE, RMSE et correlation.
    """

    print("\nValidating BG/NBD model...")

    if "actual_future_purchases" not in summary_valid.columns:
        raise ValueError(
            "Colonne actual_future_purchases manquante"
        )

    # Utiliser la meme methode de prediction corrigee
    df = predict_purchases(bgf, summary_valid, days=days)

    actual = df["actual_future_purchases"].to_numpy(dtype=float)

    predicted = df["predicted_purchases"].to_numpy(dtype=float)

    if not np.isfinite(actual).all():
        raise ValueError("Achats reels invalides")

    if not np.isfinite(predicted).all():
        raise ValueError("Predictions invalides")

    # MAE
    mae = np.mean(np.abs(actual - predicted))

    # RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # Correlation
    if np.std(actual) > 0 and np.std(predicted) > 0:
        corr = np.corrcoef(actual, predicted)[0, 1]
    else:
        corr = np.nan

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "correlation": float(corr),
        "actual_mean": float(actual.mean()),
        "pred_mean": float(predicted.mean())
    }

    print(f"\nModel Validation ({days}-day prediction):")

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Correlation: {corr:.4f}")
    print(f"Actual mean: {actual.mean():.4f}")
    print(f"Predicted mean: {predicted.mean():.4f}")

    df["actual_future_purchases_val"] = actual

    return metrics, df


##############################################################
# 7. RESUME PAR SEGMENT
##############################################################

def summarize_bgnbd_clv(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Resume les resultats CLV par segment.
    """

    segment_summary = (
        df.groupby("clv_segment", observed=True)
        .agg(
            customers=("CustomerID", "count"),
            avg_clv=("bgnbd_clv", "mean"),
            total_clv=("bgnbd_clv", "sum"),
            avg_prob_alive=("prob_alive", "mean"),
            avg_pred_purchases=("predicted_purchases", "mean")
        )
        .reset_index()
    )

    print("\nBG/NBD CLV by Segment:")

    print(segment_summary.to_string(index=False))

    return segment_summary


##############################################################
# 8. PIPELINE COMPLET
##############################################################

def run_bgnbd_pipeline(
    summary: pd.DataFrame,
    summary_valid: pd.DataFrame,
    df_train: pd.DataFrame
) -> tuple:
    """
    Execute BG/NBD, Gamma-Gamma, CLV et validation.
    """

    print("\n" + "=" * 50)
    print("BG/NBD + GAMMA-GAMMA CLV MODEL")
    print("=" * 50)

    print(f"Gross margin: {GROSS_MARGIN:.0%}")
    print(f"Discount rate: {DISCOUNT_RATE:.0%}")
    print(f"Prediction years: {PREDICTION_YEARS}")
    
    # Calculer les montants des achats repetes
    summary = calculate_repeat_monetary(
        df_train,
        summary
    )

    # 1. BG/NBD
    bgf = fit_bgnbd(summary)

    # 2. Predictions
    df_predictions = predict_purchases(
        bgf,
        summary
    )

    # 3. Gamma-Gamma
    ggf = fit_gamma_gamma(summary)

    # 4. CLV
    df_clv = predict_clv(
        bgf,
        ggf,
        df_predictions
    )

    # 5. Segmentation
    segment_summary = summarize_bgnbd_clv(df_clv)

    # 6. Validation
    # Duree reelle de la periode de validation
    validation_days = (
    summary_valid.attrs.get("validation_days", 94.18333333333334)
)

    val_metrics, val_df = validate_model(
    bgf,
    summary_valid,
    days=validation_days
)

    return (
        df_clv,
        segment_summary,
        bgf,
        ggf,
        val_metrics,
        val_df
    )
