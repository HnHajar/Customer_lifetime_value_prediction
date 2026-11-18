"""
Tests de src/segments.py   ->   pytest -q tests/test_segments.py

  1. combine_clv_models        : jointure, renommage, clients absents, colonnes manquantes
  2. assign_strategic_segments : 4 clients construits a la main (1 par quadrant), egalites, non-mutation
  3. generate_clv_summary      : contenu du rapport et coherence des chiffres
  4. run_segments_pipeline     : bout en bout sur donnees simulees

Les tests n'ont besoin d'aucun fichier de donnees : tout est simule ici.
"""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.segments import (  # noqa: E402
    combine_clv_models, assign_strategic_segments,
    generate_clv_summary, run_segments_pipeline,
)

SEGMENTS = ["Champions", "At Risk", "Rising Stars", "Dormant"]


# ── Donnees de test ──────────────────────────────────────────────────────────

def make_simple(ids, simple_clv):
    """Imite la sortie de clv_simple.calculate_simple_clv()."""
    n = len(ids)
    return pd.DataFrame({
        "CustomerID": ids,
        "simple_clv_npv": simple_clv,
        "orders_per_year": np.full(n, 4.0),
        "clv_segment": pd.Categorical(["Mid Value"] * n),
        "age_years": np.full(n, 1.0),
        "monetary": np.full(n, 50.0),          # colonne en plus, ignoree par la fusion
    })


def make_bgnbd(ids, bgnbd_clv, prob_alive=None):
    """Imite la sortie de clv_bgnbd.predict_clv()."""
    n = len(ids)
    return pd.DataFrame({
        "CustomerID": ids,
        "bgnbd_clv": bgnbd_clv,
        "prob_alive": np.full(n, 0.5) if prob_alive is None else prob_alive,
        "predicted_purchases": np.full(n, 2.0),
        "predicted_avg_value": np.full(n, 50.0),
        "frequency": np.full(n, 3.0),
        "recency": np.full(n, 100.0),
        "T": np.full(n, 200.0),
        "monetary": np.full(n, 50.0),
    })


def four_quadrants():
    """A, B, C, D : un client dans chaque quadrant de la matrice 2x2."""
    ids = ["A", "B", "C", "D"]
    simple = make_simple(ids, [100.0, 100.0, 1.0, 1.0])
    bgnbd = make_bgnbd(ids, [100.0, 1.0, 100.0, 1.0])
    return simple, bgnbd


# ── 1. combine_clv_models ────────────────────────────────────────────────────

def test_combine_merges_on_customer_id_and_keeps_expected_columns():
    simple, bgnbd = four_quadrants()
    c = combine_clv_models(simple, bgnbd)
    assert len(c) == 4
    for col in ["CustomerID", "simple_clv_npv", "orders_per_year", "age_years",
                "bgnbd_clv", "prob_alive", "predicted_purchases",
                "predicted_avg_value", "frequency", "recency", "T", "monetary"]:
        assert col in c.columns, f"Colonne manquante : {col}"


def test_combine_renames_clv_segment_to_simple_segment():
    simple, bgnbd = four_quadrants()
    c = combine_clv_models(simple, bgnbd)
    assert "simple_segment" in c.columns
    assert "clv_segment" not in c.columns


def test_combine_is_inner_join_keeps_only_common_customers():
    simple = make_simple(["A", "B", "C"], [1.0, 2.0, 3.0])
    bgnbd = make_bgnbd(["B", "C", "D"], [5.0, 6.0, 7.0])
    c = combine_clv_models(simple, bgnbd)
    assert sorted(c["CustomerID"]) == ["B", "C"]


def test_combine_values_are_aligned_per_customer():
    simple = make_simple(["A", "B"], [10.0, 20.0])
    bgnbd = make_bgnbd(["B", "A"], [200.0, 100.0])      # ordre inverse volontaire
    c = combine_clv_models(simple, bgnbd).set_index("CustomerID")
    assert c.loc["A", "simple_clv_npv"] == 10.0 and c.loc["A", "bgnbd_clv"] == 100.0
    assert c.loc["B", "simple_clv_npv"] == 20.0 and c.loc["B", "bgnbd_clv"] == 200.0


def test_combine_raises_if_a_required_column_is_missing():
    simple, bgnbd = four_quadrants()
    with pytest.raises(KeyError):
        combine_clv_models(simple.drop(columns=["simple_clv_npv"]), bgnbd)
    with pytest.raises(KeyError):
        combine_clv_models(simple, bgnbd.drop(columns=["prob_alive"]))


# ── 2. assign_strategic_segments ─────────────────────────────────────────────

def test_each_quadrant_gets_the_right_segment():
    simple, bgnbd = four_quadrants()
    out = assign_strategic_segments(combine_clv_models(simple, bgnbd)).set_index("CustomerID")
    assert out.loc["A", "strategic_segment"] == "Champions"      # simple haut, bgnbd haut
    assert out.loc["B", "strategic_segment"] == "At Risk"        # simple haut, bgnbd bas
    assert out.loc["C", "strategic_segment"] == "Rising Stars"   # simple bas,  bgnbd haut
    assert out.loc["D", "strategic_segment"] == "Dormant"        # simple bas,  bgnbd bas


def test_value_equal_to_median_counts_as_high():
    # Le code utilise >= : si tout le monde est egal a la mediane, tout le monde est "haut".
    ids = ["A", "B", "C"]
    c = combine_clv_models(make_simple(ids, [5.0] * 3), make_bgnbd(ids, [9.0] * 3))
    out = assign_strategic_segments(c)
    assert (out["strategic_segment"] == "Champions").all()


def test_assign_does_not_modify_input_and_keeps_row_count():
    simple, bgnbd = four_quadrants()
    c = combine_clv_models(simple, bgnbd)
    cols_before = list(c.columns)
    out = assign_strategic_segments(c)
    assert list(c.columns) == cols_before               # entree intacte (copy)
    assert "strategic_segment" not in c.columns
    assert len(out) == len(c)


def test_only_the_four_expected_labels_and_no_missing_value():
    rng = np.random.default_rng(1)
    n = 500
    ids = [str(i) for i in range(n)]
    c = combine_clv_models(make_simple(ids, rng.gamma(2, 100, n)),
                           make_bgnbd(ids, rng.gamma(2, 100, n)))
    out = assign_strategic_segments(c)
    assert out["strategic_segment"].notna().all()
    assert set(out["strategic_segment"]) <= set(SEGMENTS)


def test_independent_clvs_give_four_roughly_balanced_segments():
    rng = np.random.default_rng(2)
    n = 2000
    ids = [str(i) for i in range(n)]
    c = combine_clv_models(make_simple(ids, rng.gamma(2, 100, n)),
                           make_bgnbd(ids, rng.gamma(2, 100, n)))
    share = assign_strategic_segments(c)["strategic_segment"].value_counts(normalize=True)
    for s in SEGMENTS:
        assert 0.20 < share[s] < 0.30, f"{s} : {share[s]:.2%}"   # ~25 % chacun


def test_correlated_clvs_make_champions_and_dormant_dominate():
    rng = np.random.default_rng(3)
    n = 2000
    ids = [str(i) for i in range(n)]
    base = rng.gamma(2, 100, n)
    c = combine_clv_models(make_simple(ids, base),
                           make_bgnbd(ids, base * rng.uniform(0.9, 1.1, n)))
    counts = assign_strategic_segments(c)["strategic_segment"].value_counts()
    assert counts["Champions"] + counts["Dormant"] > 0.8 * n


# ── 3. generate_clv_summary ──────────────────────────────────────────────────

def _segmented():
    simple, bgnbd = four_quadrants()
    return assign_strategic_segments(combine_clv_models(simple, bgnbd))


def test_summary_is_a_string_mentioning_all_segments_and_actions():
    txt = generate_clv_summary(_segmented())
    assert isinstance(txt, str)
    for kw in ["EXECUTIVE SUMMARY", "RETAIN & REWARD", "WIN BACK",
               "NURTURE & DEVELOP", "LOW-COST REACTIVATION", "MODEL COMPARISON"]:
        assert kw in txt


def test_summary_reports_correct_totals():
    df = _segmented()
    txt = generate_clv_summary(df)
    assert "Total Customers Analyzed  : 4" in txt
    assert f"${df['simple_clv_npv'].sum():,.2f}" in txt     # 202.00
    assert f"${df['bgnbd_clv'].sum():,.2f}" in txt          # 202.00


def test_summary_segment_shares_add_up_to_100_percent():
    # 1 client par segment -> 25.0 % chacun, et chaque segment pese 100/202 etc.
    txt = generate_clv_summary(_segmented())
    assert txt.count("(25.0% of base)") == 4


def test_summary_handles_an_empty_segment():
    # Un seul client -> 3 segments sont vides. Le code ne doit pas planter.
    ids = ["A"]
    df = assign_strategic_segments(
        combine_clv_models(make_simple(ids, [10.0]), make_bgnbd(ids, [10.0])))
    try:
        txt = generate_clv_summary(df)
    except Exception as e:                      # noqa: BLE001
        pytest.fail(f"generate_clv_summary a plante avec des segments vides : {e!r}")
    assert "Total Customers Analyzed  : 1" in txt


# ── 4. run_segments_pipeline (bout en bout) ──────────────────────────────────

def test_pipeline_returns_dataframe_and_summary_and_prints(capsys):
    simple, bgnbd = four_quadrants()
    df, summary = run_segments_pipeline(simple, bgnbd)
    assert isinstance(df, pd.DataFrame) and isinstance(summary, str)
    assert len(df) == 4
    assert "strategic_segment" in df.columns            # colonne attendue par visualizations.py
    out = capsys.readouterr().out
    assert "CLV SEGMENTATION & STRATEGY" in out
    assert "Strategic Segment Distribution" in out


def test_pipeline_on_larger_simulated_data():
    rng = np.random.default_rng(4)
    n = 1500
    ids = [str(12000 + i) for i in range(n)]
    base = rng.gamma(2, 150, n)
    simple = make_simple(ids, base)
    bgnbd = make_bgnbd(ids, base * rng.uniform(0.3, 1.0, n), prob_alive=rng.random(n))
    df, _ = run_segments_pipeline(simple, bgnbd)
    assert len(df) == n
    assert df["CustomerID"].is_unique
    assert df["strategic_segment"].isin(SEGMENTS).all()