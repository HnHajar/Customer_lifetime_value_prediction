
import numpy as np
import pandas as pd

from clv_bgnbd import (
    fit_bgnbd,
    _predict_bgnbd_formula
)

# Charger les donnees deja preparees
summary = pd.read_csv("outputs/customer_summary.csv")

# Entrainer BG/NBD
bgf = fit_bgnbd(summary)

# Duree de prediction
days = 90

# Predictions de la bibliotheque lifetimes
with np.errstate(invalid="ignore"):
    standard = np.asarray(
        bgf.conditional_expected_number_of_purchases_up_to_time(
            days,
            summary["frequency"],
            summary["recency"],
            summary["T"]
        ),
        dtype=float
    )

# Predictions de notre formule
custom = np.asarray(
    _predict_bgnbd_formula(
        bgf,
        days,
        summary["frequency"],
        summary["recency"],
        summary["T"]
    ),
    dtype=float
)

# Comparer seulement les predictions standard valides
valid = np.isfinite(standard) & np.isfinite(custom)

differences = np.abs(standard[valid] - custom[valid])

print("\n=== VERIFICATION FORMULE BG/NBD ===")
print("Clients compares :", valid.sum())
print("Predictions standard invalides :", (~np.isfinite(standard)).sum())
print("Erreur moyenne :", differences.mean())
print("Erreur maximale :", differences.max())

print("\nExemples :")
for i in np.where(valid)[0][:5]:
    print(
        f"Client {summary.iloc[i]['CustomerID']} : "
        f"standard={standard[i]:.6f}, "
        f"formule={custom[i]:.6f}"
    )

# Verification de la coherence numerique
assert valid.any(), "Aucune prediction comparable"

assert np.isfinite(custom).all(), (
    "La formule contient des valeurs invalides"
)

assert (custom >= 0).all(), (
    "La formule contient des predictions negatives"
)

assert np.allclose(
    standard[valid],
    custom[valid],
    rtol=1e-5,
    atol=1e-7
), "La formule ne correspond pas aux predictions lifetimes"

print("\nTEST REUSSI : formule BG/NBD coherente")
