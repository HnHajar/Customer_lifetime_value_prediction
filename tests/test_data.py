"""Verifie la coherence des definitions dans data_prep (monetary vs repeat_monetary)."""
import os, sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_prep import build_customer_summary  # noqa: E402


def _df():
    # Client A : jour 1 = 2 factures (100 + 300) ; jour 11 = 1 facture (200)
    # Client B : un seul achat (50)
    rows = [
        ("A", "i1", "2011-01-01 09:00", 100.0), ("A", "i2", "2011-01-01 15:00", 300.0),
        ("A", "i3", "2011-01-11 10:00", 200.0), ("B", "i4", "2011-01-05 10:00", 50.0),
    ]
    d = pd.DataFrame(rows, columns=["CustomerID", "InvoiceNo", "InvoiceDate", "OrderRevenue"])
    d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"])
    d["PurchaseDay"] = d["InvoiceDate"].dt.normalize()
    return d


def test_monetary_is_per_invoice_and_consistent_with_n_orders():
    s = build_customer_summary(_df(), pd.Timestamp("2011-02-01")).set_index("CustomerID")
    a = s.loc["A"]
    assert a["n_orders"] == 3                              # factures
    assert a["monetary"] == pytest.approx((100 + 300 + 200) / 3)   # par facture = 200
    assert a["monetary"] * a["n_orders"] == pytest.approx(600)     # = CA total


def test_repeat_monetary_is_per_purchase_day_excluding_first_day():
    s = build_customer_summary(_df(), pd.Timestamp("2011-02-01")).set_index("CustomerID")
    a = s.loc["A"]
    assert a["frequency"] == 1 and a["recency"] == 10      # 2 jours d'achat distincts
    assert a["repeat_monetary"] == pytest.approx(200.0)    # seul le jour 11 compte
    assert s.loc["B", "frequency"] == 0 and s.loc["B", "repeat_monetary"] == 0
