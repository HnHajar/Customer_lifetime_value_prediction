import numpy as np
import pandas as pd
from clv_simple import (
    calculate_simple_clv, summarize_simple_clv,
    GROSS_MARGIN, DISCOUNT_RATE, PREDICTION_YEARS, DAYS_PER_YEAR
)

# TEST 1 : chargement
summary = pd.read_csv("outputs/customer_summary.csv", dtype={"CustomerID": str})
print("Clients :", len(summary))

# TEST 2 : colonnes nécessaires
for col in ["CustomerID", "T", "n_orders", "monetary"]:
    assert col in summary.columns, f"Colonne manquante : {col}"
assert (summary["T"] > 0).all()
print("TEST 2 : OK")

# TEST 3 : calcul
df = calculate_simple_clv(summary)
assert np.isfinite(df["simple_clv_npv"].to_numpy(dtype=float)).all()
assert (df["simple_clv_npv"] >= 0).all()
print("TEST 3 : OK")

# TEST 4 : vérification à la main sur le premier client
r = df.iloc[0]
age = max(r["T"] / DAYS_PER_YEAR, 0.1)
npv_factor = (1 - (1 + DISCOUNT_RATE) ** (-PREDICTION_YEARS)) / DISCOUNT_RATE
attendu = r["monetary"] * (r["n_orders"] / age) * PREDICTION_YEARS * GROSS_MARGIN * npv_factor / PREDICTION_YEARS
assert abs(attendu - r["simple_clv_npv"]) < 1e-6
print("TEST 4 : OK, client", r["CustomerID"], "CLV =", round(r["simple_clv_npv"], 2))

# TEST 5 : segments
seg = summarize_simple_clv(df)
assert seg["customers"].sum() == len(df)
assert len(seg) == 4
print("TEST 5 : OK")

print("\n=== TOUS LES TESTS TERMINÉS ===")