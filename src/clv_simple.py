import pandas as pd
import numpy as np

# les constants
GROSS_MARGIN     = 0.40    # marge brute par exemple sur 100 $ de vente, 40 $ de bénéfice
DISCOUNT_RATE    = 0.10    # taux d'actualisation annuel  10 %
PREDICTION_YEARS = 3       # horizon de prédiction 3 ans
DAYS_PER_YEAR    = 365.25  # jours par an (le .25 compense les années bissextiles)

def calculate_simple_clv(summary: pd.DataFrame,prediction_years: int = PREDICTION_YEARS) -> pd.DataFrame:

    # Calcule la Customer Lifetime Value (CLV) avec la formule simple
    # basée sur les moyennes historiques (approche RFM).
    # ─────────────────────────────────────────────────────────────────
    # FORMULE UTILISÉE
    # ─────────────────────────────────────────────────────────────────
    #     CLV = AOV × fréquence_annuelle × horizon_années × marge_brute
    # où :
    #   - AOV                = panier moyen du client (colonne "monetary")
    #   - fréquence_annuelle = nombre de commandes par an
    #   - horizon_années     = nombre d'années de projection (prediction_years)
    #   - marge_brute        = part du CA qui reste après coûts directs (constante GROSS_MARGIN)

    # On copie le DataFrame d'entrée pour NE PAS modifier `summary`
    df = summary.copy()

    # — Âge du client en années
    # La colonne "T" exprime l'âge du client en JOURS;on la convertit en ANNÉES en divisant par DAYS_PER_YEAR.
    
    df["age_years"] = df["T"] / DAYS_PER_YEAR

    # — Fréquence d'achat annualisée
    # Nombre moyen de commandes par an : n_orders (total historique) / âge en années
    df["orders_per_year"] = df["n_orders"] / df["age_years"].clip(lower=0.1)


    # — CLV brut (non actualisé)
    # Application directe de la formule : CLV = panier_moyen × commandes_par_an × horizon × marge
    # "Ce client dépense AOV $ par commande, commande N fois/an, sur H années, et chaque dollar de vente me rapporte GROSS_MARGIN de marge."
    
    df["simple_clv"] = (
        df["monetary"]          
        * df["orders_per_year"] 
        * prediction_years      
        * GROSS_MARGIN         
    )

    # — Actualisation NPV
    # L'argent reçu dans le futur vaut moins que l'argent reçu aujourd'hui
    # (inflation + coût d'opportunité). On applique donc un facteur
    # d'actualisation pour ramener les flux futurs à leur valeur actuelle

    if DISCOUNT_RATE > 0:
        npv_factor = (1 - (1 + DISCOUNT_RATE) ** (-prediction_years)) / DISCOUNT_RATE
    else:
        npv_factor = prediction_years

    df["simple_clv_npv"] = df["simple_clv"] * npv_factor / prediction_years

    # — Segmentation par quartiles
    # On veut classer ces clients en 4 groupes
    df["clv_segment"] = pd.qcut(
        df["simple_clv_npv"],
        q=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Low Value", "Mid Value", "High Value", "Top Value"]
    )
    return df
    # Le DataFrame renvoyé contient TOUTES les colonnes d'origine
    # PLUS les 5 nouvelles colonnes calculées ci-dessus.

def summarize_simple_clv(df: pd.DataFrame) -> pd.DataFrame:

    # Agrège les résultats CLV par segment et affiche un tableau de synthèse.

    # ── Agrégation par segment 
    # groupby : regroupe les clients par segment
    # observed=True : ignore les catégories vides (clv_segment est catégoriel)
    # agg syntaxe nommée : nouveau_nom = ("colonne", "fonction")
    segment_summary = (
        df.groupby("clv_segment", observed=True)
        .agg(
            customers     = ("CustomerID",     "count"),   # nb de clients
            avg_clv       = ("simple_clv_npv", "mean"),    # CLV moyen
            total_clv     = ("simple_clv_npv", "sum"),     # CLV total
            avg_aov       = ("monetary",       "mean"),    # panier moyen
            avg_frequency = ("orders_per_year","mean"),    # fréquence moyenne
        )
        .reset_index()  # transforme l'index clv_segment en colonne normale
    )

    # ── Affichage console 
    print("\n  Simple CLV Results:")
    print(f"  {'Segment':<15} {'Customers':>10} {'Avg CLV':>12} "
          f"{'Total CLV':>14} {'Avg AOV':>10}")
    print("  " + "-" * 65)
 
    # Parcours ligne par ligne 
    for _, row in segment_summary.iterrows():
       print(f"  {row['clv_segment']:<15} {int(row['customers']):>10,} "
            f"${row['avg_clv']:>11,.2f} ${row['total_clv']:>13,.2f} "
            f"${row['avg_aov']:>9,.2f}")

    # ── Total portefeuille 
    # Somme directe sur df 
    total_clv = df["simple_clv_npv"].sum()
    print(f"\n  Total portfolio CLV ({PREDICTION_YEARS}-year NPV): "
         f"${total_clv:,.2f}")

    return segment_summary
def run_simple_clv_pipeline(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Pipeline complet du modèle CLV simple.
    # Retour :
    # (df_clv, segment_summary) : tuple de deux DataFrames
    #     - df_clv          : niveau client (CLV par client)
    #     - segment_summary : niveau segment (4 lignes agrégées)

    print("\n" + "=" * 50)
    print("  SIMPLE CLV MODEL (RFM-Based)")
    print("=" * 50)

    # ── Affichage des hypothèses 
    # :.0% formate un float en pourcentage sans décimale
    print(f"  Gross margin     : {GROSS_MARGIN:.0%}")
    print(f"  Discount rate    : {DISCOUNT_RATE:.0%}")
    print(f"  Prediction years : {PREDICTION_YEARS}")

    # ── Calculs 
    # 1) CLV par client (enrichit summary de nouvelles colonnes)
    df_clv = calculate_simple_clv(summary)

    # 2) Agrégation par segment (nécessite les colonnes calculées)
    segment_summary = summarize_simple_clv(df_clv)

    # ── Retour des deux niveaux 
    return df_clv, segment_summary


if __name__ == "__main__":
    # Ce bloc ne s'exécute que si on lance ce fichier directement
    import os, sys
    # Ajoute le dossier parent du projet au chemin de recherche Python,
    # pour pouvoir faire "from src.data_prep import ..." sans installer le projet comme package.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data_prep import run_prep_pipeline
    
    # Chemin portable vers le fichier de données (relatif à ce fichier).
    # os.path.join gère le séparateur selon l'OS (\ sur Windows, / ailleurs).
    path = os.path.join(os.path.dirname(__file__), "..", "data", "OnlineRetail.xlsx")

    # Charge et prépare les données.
    # run_prep_pipeline renvoie 7 valeurs ; on ne garde que summary,
    # summary_valid et split_date (les _ ignorent le reste).
    _, _, _, summary, summary_valid, split_date, _ = run_prep_pipeline(path)

    # Lance le pipeline CLV complet (calcul + agrégation par segment).
    df_clv, seg = run_simple_clv_pipeline(summary)

    # Aperçu : 10 premiers clients avec les 5 colonnes clés.
    print(df_clv[["CustomerID", "monetary", "orders_per_year",
                  "simple_clv_npv", "clv_segment"]].head(10))