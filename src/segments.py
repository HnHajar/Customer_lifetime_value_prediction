"""
segments.py
-----------
Combines Simple CLV and BG/NBD CLV results to create actionable
customer segments with marketing recommendations.

Why combine both models?
------------------------
  - Simple CLV is based purely on historical behavior — great for
    understanding what customers have done.
  - BG/NBD CLV accounts for churn probability — great for predicting
    what customers will do next.

  Together they give a richer picture:
    High Simple CLV + High BG/NBD CLV = True Champions (reward them)
    High Simple CLV + Low BG/NBD CLV  = Valuable but at risk (win back)
    Low Simple CLV  + High BG/NBD CLV = Rising stars (nurture them)
    Low Simple CLV  + Low BG/NBD CLV  = Low priority (low-cost contact)
"""

import pandas as pd
import numpy as np


# ── Combine Models ────────────────────────────────────────────────────────────

def combine_clv_models(df_simple: pd.DataFrame, df_bgnbd: pd.DataFrame) -> pd.DataFrame:
    """
    Merge simple and BG/NBD CLV results into one customer-level DataFrame.

    Parameters
    ----------
    df_simple : pd.DataFrame  Output of clv_simple.calculate_simple_clv()
    df_bgnbd  : pd.DataFrame  Output of clv_bgnbd.predict_clv()

    Returns
    -------
    pd.DataFrame  Combined customer-level CLV data.
    """
    simple_cols = ["CustomerID", "simple_clv_npv", "orders_per_year",
                   "clv_segment", "age_years"]
    bgnbd_cols  = ["CustomerID", "bgnbd_clv", "prob_alive",
                   "predicted_purchases", "predicted_avg_value",
                   "frequency", "recency", "T", "monetary"]

    combined = (
        df_simple[simple_cols]
        .merge(df_bgnbd[bgnbd_cols], on="CustomerID", how="inner")
        .rename(columns={"clv_segment": "simple_segment"})
    )

    return combined


# ── 2x2 Strategic Segments ────────────────────────────────────────────────────

def assign_strategic_segments(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 2×2 strategic segments based on Simple CLV and BG/NBD CLV.

    Matrix
    ------
                        BG/NBD CLV
                    Low         High
    Simple    High | At Risk  | Champions |
    CLV       Low  | Dormant  | Rising    |

    Parameters
    ----------
    combined : pd.DataFrame  Output of combine_clv_models().

    Returns
    -------
    pd.DataFrame with strategic_segment column added.
    """
    df = combined.copy()

    # Median split for both CLV measures
    simple_median = df["simple_clv_npv"].median()
    bgnbd_median  = df["bgnbd_clv"].median()

    def assign_segment(row):
        high_simple = row["simple_clv_npv"] >= simple_median
        high_bgnbd  = row["bgnbd_clv"]      >= bgnbd_median

        if high_simple and high_bgnbd:
            return "Champions"
        elif high_simple and not high_bgnbd:
            return "At Risk"
        elif not high_simple and high_bgnbd:
            return "Rising Stars"
        else:
            return "Dormant"

    df["strategic_segment"] = df.apply(assign_segment, axis=1)

    return df


# ── Executive Summary ─────────────────────────────────────────────────────────

def generate_clv_summary(df: pd.DataFrame) -> str:
    """
    Generate a business-ready CLV executive summary with segment strategies.

    Parameters
    ----------
    df : pd.DataFrame  Output of assign_strategic_segments().

    Returns
    -------
    str  Formatted executive summary.
    """
    total_simple_clv = df["simple_clv_npv"].sum()
    total_bgnbd_clv  = df["bgnbd_clv"].sum()
    total_customers  = len(df)

    strategies = {
        "Champions"  : (
            "🏆 RETAIN & REWARD\n"
            "     These customers have high historical value AND high predicted future value.\n"
            "     Strategy: VIP loyalty program, early access to new products, exclusive offers.\n"
            "     Budget priority: HIGH — these customers have the highest ROI on retention spend."
        ),
        "At Risk"    : (
            "⚠️  WIN BACK\n"
            "     High historical value but model predicts they may not return.\n"
            "     Strategy: Personalized win-back campaign, time-limited discounts,\n"
            "     re-engagement email sequence. Act quickly — the longer they're gone,\n"
            "     the harder they are to recover.\n"
            "     Budget priority: HIGH — recoverable revenue is at stake."
        ),
        "Rising Stars": (
            "🌟 NURTURE & DEVELOP\n"
            "     Lower historical spend but model predicts strong future engagement.\n"
            "     Strategy: Onboarding sequence, product discovery, loyalty enrollment.\n"
            "     These customers are building habits — help them buy more.\n"
            "     Budget priority: MEDIUM — high potential, relatively low cost to develop."
        ),
        "Dormant"    : (
            "😴 LOW-COST REACTIVATION\n"
            "     Low historical value and low predicted future engagement.\n"
            "     Strategy: Broad, low-cost reactivation (e.g. email only). If no response\n"
            "     after 2 attempts, suppress from active campaigns to protect deliverability.\n"
            "     Budget priority: LOW — don't over-invest in this segment."
        ),
    }

    lines = []
    lines.append("=" * 70)
    lines.append("  CUSTOMER LIFETIME VALUE — EXECUTIVE SUMMARY")
    lines.append("  Simple CLV + BG/NBD + Gamma-Gamma | UCI Online Retail Dataset")
    lines.append("=" * 70)
    lines.append(f"\n  Total Customers Analyzed  : {total_customers:,}")
    lines.append(f"  Simple CLV Portfolio Value : ${total_simple_clv:,.2f} (3-year NPV)")
    lines.append(f"  BG/NBD CLV Portfolio Value : ${total_bgnbd_clv:,.2f} (3-year NPV)")
    lines.append("\n" + "-" * 70)

    for segment in ["Champions", "At Risk", "Rising Stars", "Dormant"]:
        seg_df   = df[df["strategic_segment"] == segment]
        n        = len(seg_df)
        pct      = n / total_customers * 100
        avg_clv  = seg_df["bgnbd_clv"].mean()
        tot_clv  = seg_df["bgnbd_clv"].sum()
        p_alive  = seg_df["prob_alive"].mean()

        lines.append(f"\n  {strategies[segment]}")
        lines.append(f"\n     Customers    : {n:,} ({pct:.1f}% of base)")
        lines.append(f"     Avg CLV      : ${avg_clv:,.2f}")
        lines.append(f"     Total CLV    : ${tot_clv:,.2f} ({tot_clv/total_bgnbd_clv:.1%} of portfolio)")
        lines.append(f"     P(Alive)     : {p_alive:.1%}")

    lines.append("\n" + "=" * 70)
    lines.append("  MODEL COMPARISON")
    lines.append("-" * 70)
    lines.append(f"\n  Simple CLV assumes all customers continue buying at their historical")
    lines.append(f"  rate indefinitely. BG/NBD accounts for churn probability, giving a")
    lines.append(f"  more conservative (and more accurate) estimate.")
    lines.append(f"\n  Simple CLV total  : ${total_simple_clv:,.2f}")
    lines.append(f"  BG/NBD CLV total  : ${total_bgnbd_clv:,.2f}")
    lines.append(f"  Difference        : ${total_simple_clv - total_bgnbd_clv:,.2f} "
                 f"({(total_simple_clv - total_bgnbd_clv)/total_simple_clv:.1%} overestimate by simple model)")
    lines.append("\n" + "=" * 70 + "\n")

    return "\n".join(lines)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_segments_pipeline(df_simple: pd.DataFrame,
                          df_bgnbd: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Full segmentation pipeline.

    Returns
    -------
    df_combined  : customer-level data with both CLV estimates and segments
    exec_summary : formatted executive summary string
    """
    print("\n" + "=" * 50)
    print("  CLV SEGMENTATION & STRATEGY")
    print("=" * 50)

    combined     = combine_clv_models(df_simple, df_bgnbd)
    df_combined  = assign_strategic_segments(combined)
    exec_summary = generate_clv_summary(df_combined)

    print(exec_summary)

    # Segment counts
    print("\n  Strategic Segment Distribution:")
    counts = df_combined["strategic_segment"].value_counts()
    for seg, n in counts.items():
        print(f"    {seg:<15}: {n:,} customers ({n/len(df_combined):.1%})")

    return df_combined, exec_summary