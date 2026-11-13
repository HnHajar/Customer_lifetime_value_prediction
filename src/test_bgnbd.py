
import pandas as pd
import numpy as np
from clv_bgnbd import calculate_repeat_monetary
from data_prep import load_data, split_data

from clv_bgnbd import (
    fit_bgnbd,
    predict_purchases,
    fit_gamma_gamma,
    predict_clv,
    validate_model,
    summarize_bgnbd_clv
)

# ==========================================
# TEST 1 : CHARGEMENT DES DONNEES
# ==========================================

summary = pd.read_csv("outputs/customer_summary.csv")

print("\n=== TEST 1 : CHARGEMENT ===")
print("Nombre de clients :", len(summary))
print("Colonnes :", summary.columns.tolist())
print(summary.head())

# ==========================================
# TEST 2 : VERIFICATION DES DONNEES
# ==========================================

required = ["frequency", "recency", "T", "monetary"]

for col in required:
    assert col in summary.columns, f"Colonne manquante : {col}"

assert (summary["frequency"] >= 0).all()
assert (summary["recency"] >= 0).all()
assert (summary["recency"] <= summary["T"]).all()
assert (summary["T"] > 0).all()

print("\nTEST 2 : Donnees valides")

# ==========================================
# TEST 3 : ENTRAINEMENT BG/NBD
# ==========================================

bgf = fit_bgnbd(summary)

print("\nTEST 3 : BG/NBD entraine")

# ==========================================
# TEST 4 : PREDICTIONS
# ==========================================

predictions = predict_purchases(
    bgf,
    summary,
    days=90
)

print("\n=== TEST 4 : PREDICTIONS ===")

print(
    predictions[
        ["frequency", "predicted_purchases", "prob_alive"]
    ].head()
)

# ==========================================
# DIAGNOSTIC DES PREDICTIONS
# ==========================================

print("\n=== DIAGNOSTIC DES PREDICTIONS ===")

nan_count = predictions["predicted_purchases"].isna().sum()

negative_count = (
    predictions["predicted_purchases"] < 0
).sum()

infinite_count = np.isinf(
    predictions["predicted_purchases"].to_numpy(dtype=float)
).sum()

print("Nombre de NaN :", nan_count)
print("Nombre de predictions negatives :", negative_count)
print("Nombre de valeurs infinies :", infinite_count)

invalid_mask = (
    ~np.isfinite(
        predictions["predicted_purchases"].to_numpy(dtype=float)
    )
    | (predictions["predicted_purchases"] < 0).to_numpy()
)

invalides = predictions.loc[invalid_mask]

if len(invalides) > 0:

    print("\nDetails des 20 premiers clients invalides :")

    print(
        invalides[
            [
                "CustomerID",
                "frequency",
                "recency",
                "T",
                "predicted_purchases"
            ]
        ].head(20).to_string(index=False)
    )

    print("\nStatistiques des clients invalides :")

    print(
        invalides[
            ["frequency", "recency", "T"]
        ].describe()
    )

    print("\nECHEC : certaines predictions sont invalides.")
    print("Il faut corriger le calcul avant de continuer.")

    raise SystemExit(1)

# Verifier les probabilites
assert np.isfinite(
    predictions["prob_alive"].to_numpy(dtype=float)
).all()

assert predictions["prob_alive"].between(0, 1).all()

print("\nTEST 4 : Predictions valides")

# ==========================================
# TEST 5 : ENTRAINEMENT GAMMA-GAMMA
# ==========================================


# ============================================================
# TEST 5 : GAMMA-GAMMA AVEC REPEAT MONETARY
# ============================================================

print("\n=== TEST 5 : GAMMA-GAMMA ===")

# 1. Charger les transactions nettoyees
df = load_data("data/OnlineRetail.xlsx")

# 2. Reproduire la separation train / validation
df_train, _, _, _ = split_data(df)

# 3. Calculer le montant moyen des achats repetes
summary = calculate_repeat_monetary(
    df_train,
    summary
)

# 4. Verifier les donnees
print("\n=== VERIFICATION REPEAT MONETARY ===")

print("Nombre de clients :", len(summary))

repeat_buyers = summary[
    summary["frequency"] > 0
]

print(
    "Clients avec achats repetes :",
    len(repeat_buyers)
)

print(
    "Montant moyen des achats repetes :",
    round(repeat_buyers["repeat_monetary"].mean(), 2)
)

print(
    "Valeurs manquantes :",
    repeat_buyers["repeat_monetary"].isna().sum()
)

# 5. Entrainer Gamma-Gamma
ggf = fit_gamma_gamma(summary)

print("\nTEST 5 : Gamma-Gamma entraine avec succes")

print("\nTEST 5 : Gamma-Gamma entraine")

# ==========================================
# TEST 6 : VALIDATION DU MODELE
# ==========================================

if "actual_future_purchases" in summary.columns:

    print("\n=== TEST 6 : VALIDATION ===")

    metrics, validation = validate_model(
        bgf,
        summary,
        days=94.18333333333334
    )

    print("Validation terminee")
    print(metrics)

else:

    print("\nTEST 6 : Validation ignoree")
    print("Colonne actual_future_purchases absente")

# ==========================================
# TEST 7 : CUSTOMER LIFETIME VALUE
# ==========================================

print("\n=== TEST 7 : CALCUL CLV ===")


# ============================================================
# TEST 7 : CALCUL CUSTOMER LIFETIME VALUE
# ============================================================

print("\n=== TEST 7 : CALCUL CLV ===")

# Recalculer les predictions avec le summary enrichi
# qui contient maintenant repeat_monetary
predictions = predict_purchases(
    bgf,
    summary,
    days=90
)

# Verifier la presence de la colonne
assert "repeat_monetary" in predictions.columns

# Calculer la CLV
clv_results = predict_clv(
    bgf,
    ggf,
    predictions
)

print("\nTEST 7 : CLV calculee avec succes")
print("Nombre de clients :", len(clv_results))
print("CLV moyenne :", round(clv_results["bgnbd_clv"].mean(), 2))
print("CLV totale :", round(clv_results["bgnbd_clv"].sum(), 2))


# Verifier les colonnes creees
assert "predicted_avg_value" in clv_results.columns
assert "bgnbd_clv" in clv_results.columns
assert "clv_segment" in clv_results.columns

# Verifier les valeurs
assert np.isfinite(
    clv_results["bgnbd_clv"].to_numpy(dtype=float)
).all()

assert (clv_results["bgnbd_clv"] >= 0).all()

print("\nTEST 7 : CLV calculee")

print("\nApercu des resultats :")
print(
    clv_results[
        [
            "CustomerID",
            "predicted_purchases",
            "predicted_avg_value",
            "bgnbd_clv",
            "clv_segment"
        ]
    ].head(10).to_string(index=False)
)

# ==========================================
# TEST 8 : SEGMENTATION CLIENTS
# ==========================================

print("\n=== TEST 8 : SEGMENTATION ===")

segment_summary = summarize_bgnbd_clv(clv_results)

assert segment_summary["customers"].sum() == len(clv_results)

print("\nTEST 8 : Segmentation valide")

# ==========================================
# FIN
# ==========================================

print("\n=== TOUS LES TESTS TERMINES ===")
