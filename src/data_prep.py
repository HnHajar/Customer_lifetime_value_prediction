#importer les bibliotheques 
import pandas as pd # pandas pour la manipulation des tabl de donnees (DataFrames)
import numpy as np  #numpy pour le calcul numerique 
import os #os pour la gestion des chemins et fichiers 

# ── Constantes ─────────────────────────────────────────────────────────────────

# Use 75% of the data to train, 25% to validate
TRAIN_FRACTION = 0.75


#fonction de chargement et nettoyage du dataset online retail.xlsx
#le fct recoit le chemin du fichier  et retourne un dataframe pandas
def load_data(filepath: str) -> pd.DataFrame:
    print(f"Loading data from: {filepath}")

    df = pd.read_excel(filepath, dtype={"CustomerID": str})
    #forcer pandas a considerer les id de clients comme des chaines de caracteres 
    print(f"Raw shape: {df.shape}")

    #nettoyage
    df = df.dropna(subset=["CustomerID"])
    #supprimer les transactions sans id client 
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    #exclure les factures annulees (les numéros de facture commençant par C correspondent aux annulations)
    #astype(str) : convertir en chaîne de caractères.
    #str.startswith("C") : détecter les factures commençant par C.
    #~ : inverser le résultat pour conserver les autres factures.
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    #exclure les quantites et prix invalides, on conserve uniquement les transactions ayant une qt strct positive et un prix unitaire strct positif

    #transformer les dates  pour pouvoir effectuer des calculs 
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    ##on cree une nouvelle colonne de revenu de chaque ligne de transaction
    df["OrderRevenue"] = df["Quantity"] * df["UnitPrice"]

    return df
#test avec cmd : (python3 -c 'from src.data_prep import load_data; df = load_data("data/raw/Online Retail.xlsx"); print(df.head())'Loading data from: data/raw/Online Retail.xlsx)
#resultat obtenu : (Raw shape: (541909, 8
  #InvoiceNo StockCode  ...         Country  OrderRevenue
#0    536365    85123A  ...  United Kingdom         15.30
#1    536365     71053  ...  United Kingdom         20.34
#2    536365    84406B  ...  United Kingdom         22.00
#3    536365    84029G  ...  United Kingdom         20.34
#4    536365    84029E  ...  United Kingdom         20.34

#[5 rows x 9 columns]
#)

# ── Train / Validation Split ──────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:

    min_date = df["InvoiceDate"].min() #on recupere la premiere date 
    max_date = df["InvoiceDate"].max() #la derniere date 
    total_days = (max_date - min_date).days # on calcul la duree totale 

    split_date = min_date + pd.Timedelta(days=int(total_days * TRAIN_FRACTION)) #calculer la date de separation (c est temporel c est pas fixe )
    end_date   = max_date
    #on separe les donnees
    df_train = df[df["InvoiceDate"] <= split_date] #on inclu les transactions du jour de separations dans training 
    df_valid  = df[df["InvoiceDate"] >  split_date]
    #resultats : 2 dataframes df_train historiques des achats pour entrainement , et df_valid pour evaluer les predictions
    print(f"\n  Train period : {min_date.date()} → {split_date.date()} ({len(df_train):,} transactions)")
    print(f"  Valid period : {split_date.date()} → {end_date.date()} ({len(df_valid):,} transactions)")

    return df_train, df_valid, split_date, end_date

#test avec cmd : (python3 -c 'from src.data_prep import load_data, split_data; df = load_data("data/OnlineRetail.xlsx"); df_train, df_valid, split_date, end_date = split_data(df); print("\nDate de séparation :", split_date); print("Fin du dataset :", end_date); print("Total :", len(df)); print("Train + Validation :", len(df_train) + len(df_valid))')
#resultats obtenu (Raw shape: (541909, 8)
  #Train period : 2010-12-01 → 2011-09-06 (231,661 transactions)
  #Valid period : 2011-09-06 → 2011-12-09 (166,223 transactions)
#Date de séparation : 2011-09-06 08:26:00
#Fin du dataset : 2011-12-09 12:50:00
#Total : 397884
#Train + Validation : 397884
#)


# ── passer des transactions individuelles à un tableau qui décrit chaque client avec des indicateurs qui va servire a alimenter les modeles clv────────────────────────────────────────────────────

def build_customer_summary(df_train: pd.DataFrame, split_date: pd.Timestamp) -> pd.DataFrame:
    #pour faire cela il faut passer de 2 agregations 
    # premierement : passer de transactions a commandes , un client peut acheter plusieurs produits dans une meme commande , or pour calculer le clv nous voulons connaitre la valeur de chaque commande pas de chaque produit 
    orders = (
        df_train.groupby(["CustomerID", "InvoiceNo", "InvoiceDate"]) # group.by on regroupe les lignes qui ont les memes idclient et numfacture et date factures , cela pour identifier chaque commande appartenant a un client 
        ["OrderRevenue"].sum() # pour chaque groupe on additionne les valeurs des produits 
        .reset_index() #remet les colonnes utilises sous forme de colonnes normales dans la dataframe df_train
    )
    #on renomme la colonne simplement 
    orders.rename(columns={"OrderRevenue": "OrderValue"}, inplace=True)

    #Résultat de la première agrégation
    #CustomerID	InvoiceNo	InvoiceDate	OrderRevenue
    #C001	F001	01/01/2021	50 €
    #C001	F002	05/01/2021	50 €
    #C002	F003	06/01/2021	15 €

    # ON A DONC UNE LIGNE PAR COMMANDE 


    # deuxiement : passer de commandes aux clients : une seule ligne par client 
    summary = (
        orders.groupby("CustomerID") #on regroupe totues les cmd appartenant au meme client 
        .agg( #agg()permet d appliquer plusieurs operations d agregation sur chaque groupe 
            first_purchase = ("InvoiceDate", "min"),
            last_purchase  = ("InvoiceDate", "max"),
            n_orders       = ("InvoiceNo",   "nunique"),#nunique compte les valeurs distinctes , le nbrs de factures differentes
            mean_order_value = ("OrderValue", "mean"),#mean calcule la moyenne
        )
        .reset_index()#meme principe , remettre customerid sous forme de colonnes 
    )

    #Résultat de la deuxième agrégation
    #CustomerID	first_purchase	last_purchase	n_orders	mean_order_value
    #C001	01/01/2021	05/01/2021	2	50 €
    #C002	06/01/2021	06/01/2021	1	15 €

    #ON A DONC UNE SEULE LIGNE PAR CLIENT 


    # et puis calculer les variables rfm+: frequency , recency, T , monetary
    summary["frequency"] = summary["n_orders"] - 1     # frequency c est le nombre d achats repetes : n_orders-1 , Pourquoi ?
    # exemple : 1 commande : frequency =0 , 2 commandes ca donne frequency=1 etccc, et donc le premier achat est considere comme l aquissition du client , on cherche a modeliser les achats repetes 
    summary["recency"]   = (summary["last_purchase"]  - summary["first_purchase"]).dt.days #recency c est le temps entre premier et dernier achat : las_purchase-first_purchase
    summary["T"]         = (split_date - summary["first_purchase"]).dt.days #duree d'observation du client : deuc clients peuvent effectue chacun 3 achats , mais sur des periodes differentes , split_date - first_purchase
    summary["monetary"]  = summary["mean_order_value"] # monetary c est la valeur moyenne d une cmd 

    # Remove customers with T = 0 (purchased only on the last day)
    summary = summary[summary["T"] > 0]

    # Remove customers with negative or zero monetary value
    summary = summary[summary["monetary"] > 0]

    print(f"\n  Customer summary table:")
    print(f"    Total customers    : {len(summary):,}")
    print(f"    One-time buyers    : {(summary['frequency'] == 0).sum():,} ({(summary['frequency'] == 0).mean():.1%})")
    print(f"    Repeat buyers      : {(summary['frequency'] > 0).sum():,} ({(summary['frequency'] > 0).mean():.1%})")
    print(f"    Avg frequency      : {summary['frequency'].mean():.2f} repeat purchases")
    print(f"    Avg monetary       : ${summary['monetary'].mean():.2f} per order")
    print(f"    Avg T (age)        : {summary['T'].mean():.0f} days")

    return summary

# test avec cmd ( python3 -c 'from src.data_prep import load_data, split_data, build_customer_summary; df = load_data("data/OnlineRetail.xlsx"); df_train, df_valid, split_date, end_date = split_data(df); summary = build_customer_summary(df_train, split_date); print("\nAperçu du tableau :"); print(summary.head()); print("\nDimensions :", summary.shape)')
#resultats obtenu (Raw shape: (541909, 8)
  #Train period : 2010-12-01 → 2011-09-06 (231,661 transactions)
  #Valid period : 2011-09-06 → 2011-12-09 (166,223 transactions)

  #Customer summary table:
    #Total customers    : 3,342           clients retenues
    #One-time buyers    : 1 368 (40.9%)   clients avec un seul achats
    #Repeat buyers      : 1 974 (59.1%)
    #Avg frequency      : 2.49 repeat purchases
    #Avg monetary       : $398.55 per order
    #Avg T (age)        : 179 days

#Aperçu du tableau :
  #CustomerID      first_purchase       last_purchase  n_orders  ...  frequency  recency    T      monetary
#0      12346 2011-01-18 10:01:00 2011-01-18 10:01:00         1  ...          0        0  230  77183.600000
#1      12347 2010-12-07 14:57:00 2011-08-02 08:48:00         5  ...          4      237  272    558.172000
#2      12348 2010-12-16 19:09:00 2011-04-05 10:47:00         3  ...          2      109  263    495.746667
#3      12350 2011-02-02 16:01:00 2011-02-02 16:01:00         1  ...          0        0  215    334.400000
#4      12352 2011-02-16 12:33:00 2011-03-22 16:08:00         5  ...          4       34  201    312.362000

#[5 rows x 9 columns]

#Dimensions : (3342, 9)

#)