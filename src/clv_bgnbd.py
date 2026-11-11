import pandas as pd
import numpy as np

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data
    LIFETIMES_AVAILABLE = True

except ImportError:
    LIFETIMES_AVAILABLE = False
    print("lifetimes library not found. Run: pip install lifetimes")

#############################################################
# CONSTANTES
#############################################################
 
GROSS_MARGIN=0.40
DISCOUNT_RATE=0.10
PREDICTION_DAYS=90
PREDICTION_YEARS=3

##############################################################
# BG/NBD
##############################################################

def fit_bgnbd(summary: pd.DataFrame) -> "BetaGeoFitter" :
    if not LIFETIMES_AVAILABLE:
        raise ImportError(
            "lifetimes library required. Run: pip install lifetimes"
        )
     
    print("\nFitting BG/NBD model...")
    bgf=BetaGeoFitter(penalizer_coef=0.01)
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

    return bgf

#############################################################
#     predict future purchases
#############################################################

def predict_purchases(
        bgf: "BetaGeoFitter",
        summary: pd.DataFrame,
        days: int = PREDICTION_DAYS

) ->pd.DataFrame:

    df=summary.copy()
    
    df["predicted_purchases"] = (
        bgf.onditional_expected_number_of_purchases_up_to_time(
            days,
            df["frequency"],
            df["recency"],
            df["T"],
        )
    )
    df["prob_alive"] = bgf.conditional_probability_alive(
        df["frequency"],
        df["recency"],
        df["T"]
    )
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
def fit_gamma_gamma(summary: pd.DataFrame) -> "GammaGammaFitter":
    if not LIFETIMES_AVAILABLE:
        raise ImportError("lifetimes library required.")

    print("\nFitting Gamma-Gamma model...")
    repeat_buyers = summary[summary["frequency"] > 0]
    print(
        f"Using {len(repeat_buyers):,} repeat buyers for fitting"
    )
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(
        repeat_buyers["frequency"],
        repeat_buyers["monetary"],
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
def predict_clv(
    bgf: "BetaGeoFitter",
    ggf: "GammaGammaFitter",
    summary: pd.DataFrame
) -> pd.DataFrame:
    df = summary.copy()
    df["pedicted_avg_value"]=(
        ggf.coconditional_expected_average_profit(
            df["frequency"],
            df["monetary"]
        )
    )
    df["bgnd_clv"] = (
         ggf.customer_lifetime_value(
             bgf,
             df["frequency"],
             df["recency"],
             df["T"],
             df["monetary"],
             time=PREDICTION_YEARS * 12,
             discount_rate=DISCOUNT_RATE / 12,
             freq="D"
             )*GROSS_MARGIN
         )
    df["clv_segment"]=pd.qcut(
        df["bgnbd_clv"],
        q=[0, 0.25, 0.50, 0.75, 1.0],
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
        f"${df['bgnbd_clv'].mean():,.2f}"
    )

    print(
        f"Total portfolio CLV: "
        f"${df['bgnbd_clv'].sum():,.2f}"
    )

    print(
        f"Median CLV: "
        f"${df['bgnbd_clv'].median():,.2f}"
    )

    return df




def validate_model(
    bgf: "BetaGeoFitter",
    summary_valid: pd.DataFrame,
    days: int = PREDICTION_DAYS
) -> dict:

    df = summary_valid.copy()

    df["predicted_purchases"] = (
        bgf.conditional_expected_number_of_purchases_up_to_time(
            days,
            df["frequency"],
            df["recency"],
            df["T"]
        )
    )

    actual = df["actual_future_purchases"].values
    predicted = df["predicted_purchases"].values

    mae = np.mean(np.abs(actual - predicted))

    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )

    corr = np.corrcoef(actual, predicted)[0, 1]

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "correlation": corr,
        "actual_mean": actual.mean(),
        "pred_mean": predicted.mean()
    }

    print(f"\nModel Validation ({days}-day prediction):")
    print(f"MAE: {mae:.4f} purchases")
    print(f"RMSE: {rmse:.4f} purchases")
    print(f"Correlation: {corr:.4f}")
    print(f"Actual mean purchases: {actual.mean():.4f}")
    print(f"Predicted mean: {predicted.mean():.4f}")

    df["actual_future_purchases_val"] = actual

    return metrics, df

###################################################################
#   MODEL VALIDATION 
###################################################################

def validate_model(
        bgf: "BetaGeoFitter",
        summary_valid: pd.DataFrame,
        days:int = PREDICTION_DAYS
    
) -> dict:
    
    df = summary_valid.copy()

    df["predicted_purchases"] = (
        bgf.conditional_expected_number_of_purchases_up_to_time(
            days,
            df["frequency"],
            df["recency"],
            df["T"]
        )
    )
    actual = df["actual_future_purchases"].values
    predicted = df["predicted_purchases"].values

    mae = np.mean(np.abs(actual - predicted))

    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )
    corr = np.corrcoef(actual, predicted)[0, 1]
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "correlation": corr,
        "actual_mean": actual.mean(),
        "pred_mean": predicted.mean()
    }

    print(f"\nModel Validation ({days}-day prediction):")
    print(f"MAE: {mae:.4f} purchases")
    print(f"RMSE: {rmse:.4f} purchases")
    print(f"Correlation: {corr:.4f}")
    print(f"Actual mean purchases: {actual.mean():.4f}")
    print(f"Predicted mean: {predicted.mean():.4f}")

    df["actual_future_purchases_val"] = actual

    return metrics, df

#########################################################################
#   SEGMENT SUMMARY
#########################################################################


def summarize_bgnbd_clv(df: pd.DataFrame) -> pd.DataFrame:

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

    print(
        f"{'Segment':<15} "
        f"{'Customers':>10} "
        f"{'Avg CLV':>12} "
        f"{'Total CLV':>14} "
        f"{'P(Alive)':>10}"
    )

    print("-" * 65)

    for _, row in segment_summary.iterrows():

        print(
            f"{row['clv_segment']:<15} "
            f"{int(row['customers']):>10,} "
            f"${row['avg_clv']:>11,.2f} "
            f"${row['total_clv']:>13,.2f} "
            f"{row['avg_prob_alive']:>9.1%}"
        )

    return segment_summary

############################################################################
# COMPLETE PIPELINE
############################################################################

def run_bgnbd_pipeline(
    summary: pd.DataFrame,
    summary_valid: pd.DataFrame
) -> tuple:

    print("\n" + "=" * 50)
    print("BG/NBD + GAMMA-GAMMA CLV MODEL")
    print("=" * 50)

    print(f"Gross margin: {GROSS_MARGIN:.0%}")
    print(f"Discount rate: {DISCOUNT_RATE:.0%}")
    print(f"Prediction years: {PREDICTION_YEARS}")

    if not LIFETIMES_AVAILABLE:
        raise ImportError(
            "Install lifetimes: pip install lifetimes"
        )

    bgf = fit_bgnbd(summary)

    df_tmp = predict_purchases(bgf, summary)

    ggf = fit_gamma_gamma(summary)

    df_clv = predict_clv(bgf, ggf, df_tmp)

    segment_summary = summarize_bgnbd_clv(df_clv)

    val_metrics, val_df = validate_model(
        bgf,
        summary_valid
    )

    return (
        df_clv,
        segment_summary,
        bgf,
        ggf,
        val_metrics,
        val_df
    )
