
import numpy as np
import pandas as pd
from lifetimes import GammaGammaFitter

from data_prep import load_data, split_data


print("=" * 55)
print("TEST GAMMA-GAMMA : MONETARY DES ACHATS REPETES")
print("=" * 55)


# ==================================================
# TEST 1 : CHARGEMENT DES DONNEES
# ==================================================

print("\n=== TEST 1 : CHARGEMENT ===")

df = load_data("data/OnlineRetail.xlsx")

df_train, df_valid, split_date, end_date = split_data(df)

original = pd.read_csv(
    "outputs/customer_summary.csv",
    dtype={"CustomerID": str}
)

print("Clients du fichier original :", len(original))
print("Transactions d'entrainement :", len(df_train))

assert not df_train.empty
assert not original.empty

print("TEST 1 : OK")


# ==================================================
# TEST 2 : CONSTRUCTION DES COMMANDES
# ==================================================

print("\n=== TEST 2 : COMMANDES ===")


orders = (
    df_train.groupby(
        ["CustomerID", "InvoiceNo"],
        as_index=False
    )
    .agg(
        InvoiceDate=("InvoiceDate", "min"),
        OrderRevenue=("OrderRevenue", "sum")
    )
)


orders = orders.sort_values(
    ["CustomerID", "InvoiceDate", "InvoiceNo"]
).copy()

print("Nombre de lignes de commandes :", len(orders))

assert not orders.empty
assert (orders["OrderRevenue"] > 0).all()

print("TEST 2 : OK")


# ==================================================
# TEST 3 : IDENTIFIER LES ACHATS REPETES
# ==================================================

print("\n=== TEST 3 : ACHATS REPETES ===")

orders["order_rank"] = (
    orders.groupby("CustomerID").cumcount()
)

# Exclure le premier achat de chaque client
repeat_orders = orders[
    orders["order_rank"] > 0
].copy()

repeat_summary = (
    repeat_orders.groupby("CustomerID")
    .agg(
        frequency=("InvoiceNo", "nunique"),
        monetary=("OrderRevenue", "mean")
    )
    .reset_index()
)

print("Clients avec achats repetes :", len(repeat_summary))

assert (repeat_summary["frequency"] > 0).all()
assert (repeat_summary["monetary"] > 0).all()

print("TEST 3 : OK")


# ==================================================
# TEST 4 : CONSERVER LES MEMES CLIENTS
# ==================================================

print("\n=== TEST 4 : COMPARAISON DES CLIENTS ===")

# Correction du probleme str / int64
repeat_summary["CustomerID"] = (
    repeat_summary["CustomerID"].astype(str)
)

original["CustomerID"] = (
    original["CustomerID"].astype(str)
)

# Clients du modele initial avec achats repetes
original_repeat = original[
    original["frequency"] > 0
].copy()

original_ids = set(original_repeat["CustomerID"])

# Conserver les clients du modele initial
repeat_summary = repeat_summary[
    repeat_summary["CustomerID"].isin(original_ids)
].copy()

new_ids = set(repeat_summary["CustomerID"])

print("Clients dans les deux :", len(original_ids & new_ids))
print("Uniquement ancien :", len(original_ids - new_ids))
print("Uniquement nouveau :", len(new_ids - original_ids))

assert original_ids == new_ids

print("TEST 4 : OK")


# ==================================================
# TEST 5 : VERIFIER LES FREQUENCES
# ==================================================

print("\n=== TEST 5 : VERIFICATION DES FREQUENCES ===")

comparison = repeat_summary.merge(
    original_repeat[
        ["CustomerID", "frequency", "monetary"]
    ],
    on="CustomerID",
    how="inner",
    suffixes=("_new", "_original"),
    validate="one_to_one"
)

differences = comparison[
    comparison["frequency_new"]
    != comparison["frequency_original"]
]

print("Clients compares :", len(comparison))
print("Frequences differentes :", len(differences))

if not differences.empty:
    print("\nExemples de differences :")
    print(
        differences[
            [
                "CustomerID",
                "frequency_new",
                "frequency_original"
            ]
        ].head(10).to_string(index=False)
    )

assert len(comparison) == len(original_repeat)

# Ce diagnostic peut reveler une difference
# dans la construction des commandes.
if differences.empty:
    print("TEST 5 : Frequences identiques")
else:
    print("TEST 5 : ATTENTION - frequences differentes")


# ==================================================
# TEST 6 : COMPARER LES MONTANTS MONETARY
# ==================================================

print("\n=== TEST 6 : COMPARAISON MONETARY ===")

print(
    "Ancienne moyenne :",
    round(comparison["monetary_original"].mean(), 2)
)

print(
    "Nouvelle moyenne :",
    round(comparison["monetary_new"].mean(), 2)
)

print("\nStatistiques des nouveaux montants :")

print(
    comparison["monetary_new"].describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    ).to_string()
)

print("TEST 6 : OK")


# ==================================================
# TEST 7 : ENTRAINER GAMMA-GAMMA
# ==================================================

print("\n=== TEST 7 : GAMMA-GAMMA ===")

ggf = GammaGammaFitter(penalizer_coef=0.01)

# Utiliser la frequence du tableau original
# pour conserver la definition du modele initial.
ggf.fit(
    comparison["frequency_original"],
    comparison["monetary_new"]
)

print("\nNouveaux parametres :")
print(ggf.params_)

print("\nAnciens parametres :")
print("p = 3.8310")
print("q = 0.3392")
print("v = 3.7103")

print("TEST 7 : Entrainement termine")


# ==================================================
# TEST 8 : VERIFIER LES PREDICTIONS
# ==================================================

print("\n=== TEST 8 : PREDICTIONS MONETAIRES ===")

predicted_value = ggf.conditional_expected_average_profit(
    comparison["frequency_original"],
    comparison["monetary_new"]
)

predicted_value = np.asarray(
    predicted_value,
    dtype=float
)

print("Nombre de predictions :", len(predicted_value))
print("NaN :", np.isnan(predicted_value).sum())
print("Valeurs infinies :", np.isinf(predicted_value).sum())
print("Valeurs negatives :", (predicted_value < 0).sum())

assert np.isfinite(predicted_value).all()
assert (predicted_value >= 0).all()

print("\nStatistiques des predictions :")

print(
    pd.Series(predicted_value).describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    ).to_string()
)

print("TEST 8 : OK")



# ==================================================
# TEST 9 : VALIDATION GAMMA-GAMMA
# ==================================================

print("\n=== TEST 9 : VALIDATION GAMMA-GAMMA ===")

# Construire les commandes de validation
valid_orders = (
    df_valid.groupby(
        ["CustomerID", "InvoiceNo"],
        as_index=False
    )
    .agg(
        OrderRevenue=("OrderRevenue", "sum")
    )
)

# Montant moyen reel par commande future
actual_monetary = (
    valid_orders.groupby("CustomerID")["OrderRevenue"]
    .mean()
    .reset_index(name="actual_monetary")
)

actual_monetary["CustomerID"] = (
    actual_monetary["CustomerID"].astype(str)
)

# Associer les predictions aux clients
predictions = comparison[
    ["CustomerID", "frequency_original", "monetary_new"]
].copy()

predictions["predicted_monetary"] = (
    ggf.conditional_expected_average_profit(
        predictions["frequency_original"],
        predictions["monetary_new"]
    )
)

# Conserver les clients ayant des achats futurs
validation = predictions.merge(
    actual_monetary,
    on="CustomerID",
    how="inner",
    validate="one_to_one"
)

print("Clients evalues :", len(validation))

assert not validation.empty

errors = (
    validation["predicted_monetary"]
    - validation["actual_monetary"]
)

mae = np.abs(errors).mean()
rmse = np.sqrt(np.mean(errors ** 2))

correlation = validation[
    ["predicted_monetary", "actual_monetary"]
].corr().iloc[0, 1]

print("MAE :", round(mae, 2), "GBP")
print("RMSE :", round(rmse, 2), "GBP")
print("Correlation :", round(correlation, 4))

print(
    "Montant moyen reel :",
    round(validation["actual_monetary"].mean(), 2),
    "GBP"
)

print(
    "Montant moyen predit :",
    round(validation["predicted_monetary"].mean(), 2),
    "GBP"
)

print("\nExemples de predictions :")
print(
    validation[
        ["CustomerID", "predicted_monetary", "actual_monetary"]
    ].head(10).to_string(index=False)
)

print("\nTEST 9 : TERMINE")

baseline_errors = (
    validation["monetary_new"]
    - validation["actual_monetary"]
)

baseline_mae = np.abs(baseline_errors).mean()
baseline_rmse = np.sqrt(np.mean(baseline_errors ** 2))

print("\n=== COMPARAISON AVEC MODELE SIMPLE ===")
print("MAE Gamma-Gamma :", round(mae, 2), "GBP")
print("MAE Baseline :", round(baseline_mae, 2), "GBP")
print("RMSE Gamma-Gamma :", round(rmse, 2), "GBP")
print("RMSE Baseline :", round(baseline_rmse, 2), "GBP")


# ==================================================
# TEST 10 : HYPOTHESE D'INDEPENDANCE
# ==================================================

print("\n=== TEST 10 : INDEPENDANCE GAMMA-GAMMA ===")

from scipy.stats import spearmanr

correlation_spearman, p_value = spearmanr(
    comparison["frequency_original"],
    comparison["monetary_new"]
)

print("Correlation Spearman :", round(correlation_spearman, 4))
print("P-value :", round(p_value, 6))

if abs(correlation_spearman) < 0.1:
    print("Dependance monotone faible")
else:
    print("Dependance monotone a examiner")

print("TEST 10 : TERMINE")


# ==================================================
# BILAN FINAL
# ==================================================

print("\n" + "=" * 55)
print("BILAN FINAL GAMMA-GAMMA")
print("=" * 55)

print("Clients analyses :", len(comparison))
print("Frequences differentes :", len(differences))
print("Nouveau q :", round(float(ggf.params_["q"]), 6))

if ggf.params_["q"] <= 1:
    print(
        "ATTENTION : q <= 1, la moyenne monetaire "
        "theorique de la population n'est pas finie."
    )

print("\n=== TESTS TERMINES ===")
