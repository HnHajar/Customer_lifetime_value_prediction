"""
API + dashboard pour le projet Customer Lifetime Value.

Lancement (depuis la racine du projet) :
    uvicorn app.server:app --reload --port 8000
puis ouvrir http://localhost:8000

Le code de src/ n'est PAS modifie : ce serveur appelle les memes fonctions que main.py.
"""
import io
import os
import sys
import threading
import traceback
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src import clv_bgnbd as bg
from src import clv_simple as sm
from src.data_prep import run_prep_pipeline
from src.visualizations import build_df_combined

DEFAULT_DATA = ROOT / "data" / "OnlineRetail.xlsx"
UPLOAD_PATH = ROOT / "data" / "uploaded.xlsx"
OUT_DIR = ROOT / "outputs"
STATIC = Path(__file__).parent / "static"

DEFAULTS = {"gross_margin": 0.40, "discount_rate": 0.10, "years": 3}
SEGMENTS = ["Champions", "At Risk", "Rising Stars", "Dormant"]

app = FastAPI(title="CLV Dashboard API")
LOCK = threading.Lock()
S = {  # etat du serveur (un seul utilisateur / une seule analyse a la fois)
    "status": "idle", "step": "", "progress": 0, "log": [], "error": None,
    "params": dict(DEFAULTS), "source": None,
    "df": None, "val": None, "metrics": None,
    "summary": None, "summary_valid": None, "bgf": None, "ggf": None,
}


class _Log(io.TextIOBase):
    def write(self, s):
        for line in s.splitlines():
            if line.strip():
                S["log"].append(line.rstrip())
        return len(s)


def _step(name, pct):
    S["step"], S["progress"] = name, pct


def _compute(params):
    """Recalcule les deux CLV avec les parametres donnes (modeles deja ajustes)."""
    summary = S["summary"]
    # clv_simple lit ses constantes au niveau du module : on les ajuste le temps du calcul
    sm.GROSS_MARGIN, sm.DISCOUNT_RATE = params["gross_margin"], params["discount_rate"]
    df_simple = sm.calculate_simple_clv(summary, prediction_years=params["years"])
    df_tmp = bg.predict_purchases(S["bgf"], bg._prepare_summary(summary))
    clv = bg.predict_clv(S["bgf"], S["ggf"], df_tmp, years=params["years"],
                         gross_margin=params["gross_margin"],
                         annual_discount=params["discount_rate"])
    df = build_df_combined(clv, df_simple)
    for c in ("first_purchase", "last_purchase"):
        df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    df["CustomerID"] = df["CustomerID"].astype(str)
    S["df"], S["params"] = df, dict(params)


def _run_full(path, params):
    try:
        with LOCK, redirect_stdout(_Log()):
            S.update(status="running", error=None, log=[], df=None)
            _step("Preparation des donnees", 8)
            (_, _, _, summary, summary_valid, split_date, end_date) = run_prep_pipeline(str(path))
            summary_valid.attrs["validation_days"] = (end_date - split_date).total_seconds() / 86400
            S["summary"], S["summary_valid"] = summary, summary_valid

            _step("Ajustement BG/NBD (peut prendre 1-2 min)", 30)
            bgf = bg.fit_bgnbd(summary)
            _step("Ajustement Gamma-Gamma", 65)
            ggf = bg.fit_gamma_gamma(summary)
            S["bgf"], S["ggf"] = bgf, ggf

            _step("Calcul des CLV et des segments", 80)
            _compute(params)
            _step("Validation sur la periode de test", 92)
            S["metrics"], S["val"] = bg.validate_model(bgf, summary_valid, ggf=ggf)

            OUT_DIR.mkdir(exist_ok=True)
            S["df"].to_csv(OUT_DIR / "dashboard_customers.csv", index=False)
            S["source"] = Path(path).name
            _step("Termine", 100)
            S["status"] = "done"
    except Exception as e:  # noqa
        S["status"], S["error"] = "error", f"{type(e).__name__}: {e}"
        S["log"].append(traceback.format_exc().splitlines()[-1])


# ── helpers ──────────────────────────────────────────────────────────────────
def _need_data():
    if S["df"] is None:
        raise HTTPException(409, "Aucune analyse disponible : lancez le traitement.")
    return S["df"]


def _f(x, nd=2):
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _subset(segments: str | None):
    df = _need_data()
    if segments:
        wanted = [s for s in segments.split(",") if s]
        df = df[df["strategic_segment"].isin(wanted)]
    return df


# ── routes ───────────────────────────────────────────────────────────────────
@app.get("/api/state")
def state():
    return {k: S[k] for k in ("status", "step", "progress", "error", "params", "source")} | {
        "log": S["log"][-40:], "has_data": S["df"] is not None,
        "default_data_exists": DEFAULT_DATA.exists(), "defaults": DEFAULTS,
    }


@app.post("/api/run")
async def run(file: UploadFile | None = File(None),
              gross_margin: float = Form(DEFAULTS["gross_margin"]),
              discount_rate: float = Form(DEFAULTS["discount_rate"]),
              years: int = Form(DEFAULTS["years"])):
    if S["status"] == "running":
        raise HTTPException(409, "Un traitement est deja en cours.")
    path = DEFAULT_DATA
    if file is not None and file.filename:
        if not file.filename.lower().endswith((".xlsx", ".xls")):
            raise HTTPException(400, "Format attendu : fichier Excel (.xlsx) au format Online Retail.")
        UPLOAD_PATH.parent.mkdir(exist_ok=True)
        UPLOAD_PATH.write_bytes(await file.read())
        path = UPLOAD_PATH
    if not path.exists():
        raise HTTPException(400, f"Fichier introuvable : {path.relative_to(ROOT)}. Importez OnlineRetail.xlsx.")
    params = {"gross_margin": gross_margin, "discount_rate": discount_rate, "years": years}
    S["status"] = "running"
    threading.Thread(target=_run_full, args=(path, params), daemon=True).start()
    return {"ok": True}


@app.post("/api/recompute")
def recompute(gross_margin: float = Form(...), discount_rate: float = Form(...), years: int = Form(...)):
    """Nouveaux parametres sans re-ajuster les modeles (quasi instantane)."""
    if S["bgf"] is None:
        raise HTTPException(409, "Lancez d'abord un traitement complet.")
    with LOCK, redirect_stdout(_Log()):
        _compute({"gross_margin": gross_margin, "discount_rate": discount_rate, "years": years})
    return {"ok": True}


@app.get("/api/overview")
def overview(segments: str | None = None):
    full = _need_data()
    df = _subset(segments)
    if df.empty:
        raise HTTPException(404, "Aucun client pour ce filtre.")

    clv = df["bgnbd_clv"]
    hi = float(clv.quantile(0.99))
    counts, edges = np.histogram(clv.clip(upper=hi), bins=24, range=(0, max(hi, 1e-9)))
    a_counts, _ = np.histogram(df["prob_alive"], bins=10, range=(0, 1))

    seg = (full.groupby("strategic_segment")
           .agg(customers=("CustomerID", "count"), total_clv=("bgnbd_clv", "sum"),
                avg_clv=("bgnbd_clv", "mean"), avg_simple=("simple_clv_npv", "mean"),
                prob_alive=("prob_alive", "mean"), avg_monetary=("monetary", "mean"),
                avg_freq=("frequency", "mean")).reindex(SEGMENTS).fillna(0))
    segments_out = [{"segment": k, **{c: _f(v, 4) for c, v in r.items()}} for k, r in seg.iterrows()]

    # calibration : prevu vs reel par decile de prevision (periode de validation)
    calib = []
    if S["val"] is not None:
        v = S["val"].copy()
        v["bin"] = pd.qcut(v["predicted_purchases"].rank(method="first"), 10, labels=False)
        g = v.groupby("bin").agg(pred=("predicted_purchases", "mean"),
                                 actual=("actual_future_purchases", "mean"))
        calib = [{"pred": _f(r.pred, 3), "actual": _f(r.actual, 3)} for r in g.itertuples()]

    sample = df.sample(min(len(df), 1500), random_state=0)
    top = df.nlargest(10, "bgnbd_clv")
    return {
        "params": S["params"], "source": S["source"],
        "kpis": {
            "customers": int(len(df)), "total_clv": _f(clv.sum()), "avg_clv": _f(clv.mean()),
            "median_clv": _f(clv.median()), "total_simple": _f(df["simple_clv_npv"].sum()),
            "prob_alive": _f(df["prob_alive"].mean(), 4),
            "top10_share": _f(top["bgnbd_clv"].sum() / max(clv.sum(), 1e-9), 4),
            "repeat_share": _f((df["frequency"] > 0).mean(), 4),
        },
        "segments": segments_out,
        "hist": {"edges": [_f(e) for e in edges], "counts": counts.tolist(), "clipped_at": _f(hi)},
        "alive_hist": a_counts.tolist(),
        "scatter": [{"id": r.CustomerID, "x": _f(r.simple_clv_npv), "y": _f(r.bgnbd_clv),
                     "s": r.strategic_segment} for r in sample.itertuples()],
        "top": [{"id": r.CustomerID, "clv": _f(r.bgnbd_clv), "alive": _f(r.prob_alive, 3),
                 "segment": r.strategic_segment} for r in top.itertuples()],
        "metrics": {k: _f(v, 4) for k, v in (S["metrics"] or {}).items()},
        "calibration": calib,
    }


COLS = ["CustomerID", "strategic_segment", "bgnbd_clv", "simple_clv_npv", "prob_alive",
        "frequency", "recency", "T", "monetary", "predicted_purchases"]


@app.get("/api/customers")
def customers(q: str = "", segments: str | None = None, sort: str = "bgnbd_clv",
              order: str = "desc", page: int = 1, size: int = 15):
    df = _subset(segments)
    if q:
        df = df[df["CustomerID"].str.contains(q.strip(), case=False, regex=False)]
    if sort not in COLS:
        sort = "bgnbd_clv"
    df = df.sort_values(sort, ascending=(order == "asc"))
    total = len(df)
    part = df.iloc[(page - 1) * size: page * size][COLS]
    rows = [{c: (v if isinstance(v, str) else _f(v, 4)) for c, v in r.items()}
            for r in part.to_dict("records")]
    return {"total": total, "rows": rows}


@app.get("/api/customer/{cid}")
def customer(cid: str):
    df = _need_data()
    r = df[df["CustomerID"] == cid]
    if r.empty:
        raise HTTPException(404, "Client introuvable.")
    r = r.iloc[0]
    x, tx, T = (np.array([float(r[c])]) for c in ("frequency", "recency", "T"))
    curve = [0.0] + [_f(S["bgf"].expected_purchases(m * bg.DAYS_PER_MONTH, x, tx, T)[0], 3)
                     for m in range(1, 13)]
    info = {k: (v if isinstance(v, str) else _f(v, 4)) for k, v in r.items()
            if k in COLS + ["first_purchase", "last_purchase", "n_orders", "total_revenue",
                            "predicted_avg_value", "orders_per_year"]}
    return {"info": info, "curve": curve}


@app.get("/api/export.csv")
def export(segments: str | None = None):
    df = _subset(segments)
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=clv_clients.csv"})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
