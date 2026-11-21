from pathlib import Path

import pandas as pd

from src.data_prep import run_prep_pipeline
from src.clv_bgnbd import run_bgnbd_pipeline
from src.clv_simple import run_simple_clv_pipeline
from src.visualizations import generate_all_plots, build_df_combined

def main():
    # Chemin du projet
    project_dir = Path(__file__).resolve().parent

    # Chemin du dataset
    data_path = project_dir / "data" / "OnlineRetail.xlsx"

    # Dossier pour enregistrer les resultats
    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n====================================")
    print(" CUSTOMER LIFETIME VALUE PREDICTION")
    print("====================================")

    # ETAPE 1 : Preparation des donnees
    (
        df,
        df_train,
        df_valid,
        summary,
        summary_valid,
        split_date,
        end_date
    ) = run_prep_pipeline(str(data_path))

    summary_valid.attrs["validation_days"] = (
        end_date - split_date
    ).total_seconds() / 86400

    # ETAPE 1B : CLV simple
    df_clv_simple, simple_segments = run_simple_clv_pipeline(summary)

    simple_path = output_dir / "simple_clv_predictions.csv"
    simple_segments_path = output_dir / "simple_clv_segments.csv"
    df_clv_simple.to_csv(simple_path, index=False)
    simple_segments.to_csv(simple_segments_path, index=False)

    # ETAPE 2 : BG/NBD + Gamma-Gamma
    (
        clv_results,
        segment_summary,
        bgf,
        ggf,
        validation_metrics,
        validation_results
    ) = run_bgnbd_pipeline(
        summary,
        summary_valid
    )

    # ETAPE 3 : Enregistrer les predictions CLV
    clv_path = output_dir / "bgnbd_clv_predictions.csv"
    clv_results.to_csv(clv_path, index=False)

    # ETAPE 4 : Enregistrer la segmentation
    segment_path = output_dir / "bgnbd_segments.csv"
    segment_summary.to_csv(segment_path, index=False)

    # ETAPE 5 : Enregistrer la validation
    metrics_path = output_dir / "bgnbd_validation_metrics.csv"
    pd.DataFrame([validation_metrics]).to_csv(metrics_path, index=False)

    # ETAPE 6 : Enregistrer les predictions de validation
    validation_path = output_dir / "bgnbd_validation_predictions.csv"
    validation_results.to_csv(validation_path, index=False)

    print("\n====================================")
    print(" PIPELINE TERMINE")
    print("====================================")

    print("\nResultats sauvegardes dans outputs/")
    print("-", simple_path.name)
    print("-", simple_segments_path.name)
    print("-", clv_path.name)
    print("-", segment_path.name)
    print("-", metrics_path.name)
    print("-", validation_path.name)

    print("\nNombre de clients :", len(clv_results))
    print(
        "CLV moyenne :",
        round(clv_results["bgnbd_clv"].mean(), 2)
    )
    print(
        "CLV totale :",
        round(clv_results["bgnbd_clv"].sum(), 2)
    )

    print("\nMetriques de validation :")
    for name, value in validation_metrics.items():
        print(f"{name}: {value:.4f}")

    #visualization
    print("\n[5/5] Generating Visualizations")
    df_combined = build_df_combined(clv_results, df_clv_simple)
    generate_all_plots(df_combined, validation_results, validation_metrics)



if __name__ == "__main__":
    main()