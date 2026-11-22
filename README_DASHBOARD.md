# Tableau de bord CLV

Interface web connectée au pipeline existant (`src/`, inchangé).

## Installation et lancement
```bash
pip install -r requirements-app.txt
# placer le dataset dans data/OnlineRetail.xlsx (ou l'importer depuis l'interface)
python start_dashboard.py          # ou : uvicorn app.server:app --port 8000
```
Ouvrir http://localhost:8000, puis « Lancer le traitement ».

## Ce que fait l'interface
- Lance `run_prep_pipeline` + ajustement BG/NBD / Gamma-Gamma + validation (suivi en direct).
- Marge brute, taux d'actualisation et horizon : recalcul des CLV **sans ré-ajuster les modèles**.
- Filtre par segment stratégique, matrice CLV simple / CLV BG/NBD, tableau clients (tri, recherche, export CSV), fiche client avec projection à 12 mois, fiabilité du modèle.

## Architecture
- `app/server.py` : API FastAPI (/api/run, /api/recompute, /api/overview, /api/customers, /api/customer/{id}, /api/export.csv).
- `app/static/index.html` : dashboard (Chart.js embarqué dans `app/static/vendor`, fonctionne hors ligne).
- Remarque : `requiremnets.txt` est mal orthographié dans le dépôt d'origine (le README référence `requirements.txt`).
