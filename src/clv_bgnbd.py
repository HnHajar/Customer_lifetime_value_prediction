"""
clv_bgnbd.py
------------
BG/NBD + Gamma-Gamma, implementation autonome (numpy + scipy uniquement).

Pourquoi ne plus dependre de `lifetimes` ?
  * la librairie n'est plus maintenue et casse regulierement avec les
    nouvelles versions de numpy / pandas / scipy ;
  * elle peut renvoyer des NaN (a <= 1) sans le signaler ;
  * son optimiseur part d'un seul point initial.

Ce module est valide numeriquement contre `lifetimes` (voir tests/).

Robustesse
  * calculs en espace logarithmique (gammaln, logaddexp, expit) ;
  * parametres optimises en log-espace avec bornes -> toujours > 0 ;
  * multi-start (plusieurs points initiaux), on garde le meilleur optimum ;
  * penalisation L2 choisie par validation croisee (regle du 1-ecart-type) ;
  * Gamma-Gamma : q > 1 impose (sinon la moyenne n'existe pas) ;
  * montants ecretes (winsorisation) : les grossistes d'Online Retail
    faussent sinon l'estimation de Gamma-Gamma ;
  * verifications explicites : aucune valeur NaN ne peut sortir en silence.

Entrees attendues (summary) -- une ligne par client :
  frequency        nombre d'ACHATS REPETES (jours d'achat distincts - 1)
  recency          jours entre premier et dernier achat
  T                jours entre premier achat et fin d'observation
  repeat_monetary  panier moyen des achats repetes (premier achat exclu)
                   [a defaut, `monetary` est utilise avec un avertissement]

API identique a l'ancienne version :
  run_bgnbd_pipeline(summary, summary_valid)
      -> df_clv, segment_summary, bgf, ggf, val_metrics, val_df
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln, hyp2f1
from scipy.stats import spearmanr

# -- Constantes ---------------------------------------------------------------

GROSS_MARGIN      = 0.40      # marge brute
DISCOUNT_RATE     = 0.10      # taux d'actualisation ANNUEL
PREDICTION_DAYS   = 90        # horizon "court terme" par defaut
PREDICTION_YEARS  = 3         # horizon CLV
DAYS_PER_MONTH    = 365.25 / 12

MONETARY_CLIP_Q   = 0.995     # ecretage des paniers extremes (None = desactive)
PENALIZER_GRID    = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
CV_FOLDS          = 5
RANDOM_STATE      = 42

_EPS = 1e-12


# =============================================================================
# 1. BG/NBD
# =============================================================================

def bgnbd_log_likelihood(params, x, tx, T):
    """
    Log-vraisemblance BG/NBD par client (Fader, Hardie & Lee, 2005).

    params = (r, alpha, a, b)
    x = achats repetes, tx = recency, T = age.
    """
    r, alpha, a, b = params
    A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    A2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    A3 = -(r + x) * np.log(alpha + T)
    ll = A1 + A2 + A3

    rep = x > 0
    if rep.any():
        # b + x - 1 >= b > 0 pour x >= 1 : pas de log d'un terme negatif
        A4 = (np.log(a) - np.log(b + x[rep] - 1.0)
              - (r + x[rep]) * np.log(alpha + tx[rep]))
        ll[rep] += np.logaddexp(0.0, A4 - A3[rep])
    return ll


class BGNBDModel:
    """BG/NBD : nombre d'achats futurs et probabilite d'etre encore actif."""

    PARAM_NAMES = ("r", "alpha", "a", "b")
    _LOG_BOUNDS = (
        (np.log(1e-3), np.log(1e3)),   # r
        (np.log(1e-3), np.log(1e5)),   # alpha
        (np.log(1e-3), np.log(1e3)),   # a
        (np.log(1e-3), np.log(1e3)),   # b
    )

    def __init__(self, penalizer=0.0, n_starts=8, random_state=RANDOM_STATE):
        self.penalizer = float(penalizer)
        self.n_starts = int(n_starts)
        self.random_state = random_state
        self.params_ = None
        self.converged_ = False
        self.loglik_ = np.nan
        self.n_obs_ = 0
        self.bounds_hit_ = []

    # ---- ajustement ---------------------------------------------------------

    def _neg_obj(self, theta, x, tx, T):
        ll = bgnbd_log_likelihood(np.exp(theta), x, tx, T)
        val = -np.mean(ll) + self.penalizer * np.sum(theta ** 2)
        return val if np.isfinite(val) else 1e10

    def _starts(self, T):
        mT = max(float(np.mean(T)), 1.0)
        base = np.log(np.array([
            [1.0, 0.30 * mT, 1.0, 1.0],
            [0.5, 0.10 * mT, 0.8, 2.0],
            [1.0, 1.00 * mT, 2.0, 5.0],
            [2.0, 0.50 * mT, 1.0, 3.0],
            [0.3, 0.05 * mT, 0.5, 1.5],
        ]))
        rng = np.random.default_rng(self.random_state)
        starts = list(base)
        while len(starts) < self.n_starts:
            starts.append(base[rng.integers(len(base))] + rng.normal(0, 0.5, 4))
        lo = np.array([b[0] for b in self._LOG_BOUNDS])
        hi = np.array([b[1] for b in self._LOG_BOUNDS])
        return [np.clip(s, lo, hi) for s in starts[: self.n_starts]]

    def fit(self, frequency, recency, T):
        x, tx, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        _check_rfm(x, tx, T)

        best = None
        for s in self._starts(T):
            try:
                res = minimize(self._neg_obj, s, args=(x, tx, T),
                               method="L-BFGS-B", bounds=self._LOG_BOUNDS)
            except Exception:                      # un depart rate ne bloque pas
                continue
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
        if best is None:
            raise RuntimeError("BG/NBD : l'optimisation a echoue sur tous les departs.")

        self.params_ = dict(zip(self.PARAM_NAMES, np.exp(best.x)))
        self.converged_ = bool(best.success)
        self.loglik_ = float(np.sum(bgnbd_log_likelihood(np.exp(best.x), x, tx, T)))
        self.n_obs_ = len(x)
        self.bounds_hit_ = [
            n for n, v, (lo, hi) in zip(self.PARAM_NAMES, best.x, self._LOG_BOUNDS)
            if v <= lo + 1e-6 or v >= hi - 1e-6
        ]
        return self

    # ---- predictions --------------------------------------------------------

    def _p(self):
        if self.params_ is None:
            raise RuntimeError("Modele non ajuste.")
        p = self.params_
        a = p["a"]
        if abs(a - 1.0) < 1e-4:                    # singularite (a - 1) au denominateur
            a = 1.0 + 1e-4 * (1.0 if a >= 1.0 else -1.0)
        return p["r"], p["alpha"], a, p["b"]

    def _log_dropout_odds(self, x, tx, T):
        """log( P(mort) / P(vivant) ) pour x > 0 ; -inf pour x = 0."""
        r, alpha, a, b = self._p()
        out = np.full_like(x, -np.inf, dtype=float)
        rep = x > 0
        out[rep] = (np.log(a) - np.log(b + x[rep] - 1.0)
                    + (r + x[rep]) * (np.log(alpha + T[rep]) - np.log(alpha + tx[rep])))
        return out

    def prob_alive(self, frequency, recency, T):
        x, tx, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        p = expit(-self._log_dropout_odds(x, tx, T))   # x = 0 -> expit(inf) = 1
        return np.clip(p, 0.0, 1.0)

    def expected_purchases(self, t, frequency, recency, T):
        """E[nombre d'achats dans les t prochains jours | x, tx, T]."""
        r, alpha, a, b = self._p()
        x, tx, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        t = float(t)
        if t <= 0:
            return np.zeros_like(x)

        z = t / (alpha + T + t)
        log_pow = (r + x) * (np.log(alpha + T) - np.log(alpha + T + t))
        hyp = hyp2f1(r + x, b + x, a + b + x - 1.0, z)
        term = 1.0 - np.exp(log_pow) * hyp
        # Pour a < 1 le coefficient ET le terme sont negatifs : le produit reste
        # positif. On n'ecrete donc QUE le produit (arrondis ~ -1e-16).
        numer = (a + b + x - 1.0) / (a - 1.0) * term

        log_odds = np.clip(self._log_dropout_odds(x, tx, T), -700, 700)
        denom = 1.0 + np.exp(log_odds)             # x = 0 -> exp(-inf) = 0 -> denom = 1
        out = numer / denom

        if not np.isfinite(out).all():
            bad = int((~np.isfinite(out)).sum())
            raise ValueError(
                f"BG/NBD : {bad} predictions non finies "
                f"(parametres : {self.params_}). Verifier les donnees d'entree."
            )
        return np.maximum(out, 0.0)


# =============================================================================
# 2. Gamma-Gamma
# =============================================================================

def gamma_gamma_log_likelihood(params, x, m):
    """Log-vraisemblance Gamma-Gamma (Fader & Hardie, 2013), ponderee par x."""
    p, q, v = params
    px = p * x
    return (gammaln(px + q) - gammaln(px) - gammaln(q)
            + q * np.log(v) + (px - 1.0) * np.log(m)
            + px * np.log(x) - (px + q) * np.log(v + m * x))


class GammaGammaModel:
    """Gamma-Gamma : valeur moyenne attendue d'un achat futur."""

    PARAM_NAMES = ("p", "q", "v")
    # parametrage (log p, log(q-1), log v) -> q > 1 garanti
    _LOG_BOUNDS = (
        (np.log(1e-2), np.log(1e3)),   # p
        (np.log(5e-2), np.log(1e3)),   # q - 1   (q >= 1.05)
        (np.log(1e-2), np.log(1e6)),   # v
    )

    def __init__(self, penalizer=0.0, n_starts=6, random_state=RANDOM_STATE):
        self.penalizer = float(penalizer)
        self.n_starts = int(n_starts)
        self.random_state = random_state
        self.params_ = None
        self.clip_value_ = np.inf
        self.converged_ = False
        self.loglik_ = np.nan
        self.n_obs_ = 0
        self.bounds_hit_ = []

    @staticmethod
    def _to_params(theta):
        return np.array([np.exp(theta[0]), 1.0 + np.exp(theta[1]), np.exp(theta[2])])

    def _neg_obj(self, theta, x, m):
        ll = gamma_gamma_log_likelihood(self._to_params(theta), x, m)
        val = -np.mean(ll) + self.penalizer * np.sum(theta ** 2)
        return val if np.isfinite(val) else 1e10

    def fit(self, frequency, monetary, clip_quantile=MONETARY_CLIP_Q):
        x = np.asarray(frequency, dtype=float)
        m = np.asarray(monetary, dtype=float)
        ok = np.isfinite(x) & np.isfinite(m) & (x > 0) & (m > 0)
        x, m = x[ok], m[ok]
        if len(x) < 30:
            raise ValueError(f"Gamma-Gamma : seulement {len(x)} clients repetitifs valides (min 30).")

        self.clip_value_ = float(np.quantile(m, clip_quantile)) if clip_quantile else np.inf
        m = np.minimum(m, self.clip_value_)

        mean_m = float(np.mean(m))
        rng = np.random.default_rng(self.random_state)
        # point de depart : q=2 -> E[Z] = p*v ; on choisit v pour retrouver la moyenne
        base = [(1.0, 2.0), (2.0, 3.0), (0.5, 1.5), (3.0, 5.0), (1.0, 10.0), (5.0, 2.0)]
        starts = []
        for p0, q0 in base[: self.n_starts]:
            v0 = mean_m * (q0 - 1.0) / p0
            starts.append(np.array([np.log(p0), np.log(q0 - 1.0), np.log(v0)]))
        while len(starts) < self.n_starts:
            starts.append(starts[rng.integers(len(base))] + rng.normal(0, 0.3, 3))
        lo = np.array([b[0] for b in self._LOG_BOUNDS])
        hi = np.array([b[1] for b in self._LOG_BOUNDS])

        best = None
        for s in starts:
            try:
                res = minimize(self._neg_obj, np.clip(s, lo, hi), args=(x, m),
                               method="L-BFGS-B", bounds=self._LOG_BOUNDS)
            except Exception:
                continue
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
        if best is None:
            raise RuntimeError("Gamma-Gamma : l'optimisation a echoue sur tous les departs.")

        params = self._to_params(best.x)
        self.params_ = dict(zip(self.PARAM_NAMES, params))
        self.converged_ = bool(best.success)
        self.loglik_ = float(np.sum(gamma_gamma_log_likelihood(params, x, m)))
        self.n_obs_ = len(x)
        self.bounds_hit_ = [
            n for n, v, (l, h) in zip(self.PARAM_NAMES, best.x, self._LOG_BOUNDS)
            if v <= l + 1e-6 or v >= h - 1e-6
        ]
        return self

    def expected_average_value(self, frequency, monetary):
        """
        E[valeur d'un achat futur | x, m]  =  p (v + x m) / (p x + q - 1).
        Pour x = 0 (aucun achat repete) : moyenne de la population p v / (q - 1).
        """
        if self.params_ is None:
            raise RuntimeError("Modele non ajuste.")
        p, q, v = (self.params_[k] for k in self.PARAM_NAMES)
        x = np.asarray(frequency, dtype=float)
        m = np.nan_to_num(np.asarray(monetary, dtype=float), nan=0.0)
        m = np.minimum(m, self.clip_value_)
        x_eff = np.where(m > 0, x, 0.0)            # pas de montant -> prior seul
        out = p * (v + x_eff * m) / (p * x_eff + q - 1.0)
        if not np.isfinite(out).all():
            raise ValueError("Gamma-Gamma : valeurs attendues non finies.")
        return out


# =============================================================================
# 3. Utilitaires : validation des donnees, validation croisee
# =============================================================================

def _check_rfm(x, tx, T):
    if not (np.isfinite(x).all() and np.isfinite(tx).all() and np.isfinite(T).all()):
        raise ValueError("frequency / recency / T contiennent des NaN ou inf.")
    if (x < 0).any() or (tx < 0).any():
        raise ValueError("frequency et recency doivent etre >= 0.")
    if (T <= 0).any():
        raise ValueError("T doit etre > 0 (retirer les clients sans duree d'observation).")
    if (tx > T + 1e-9).any():
        raise ValueError("recency > T pour au moins un client.")
    if ((x > 0) & (tx <= 0)).any():
        raise ValueError(
            "frequency > 0 avec recency = 0 : les achats doivent etre comptes par JOUR "
            "distinct (voir data_prep.build_customer_summary)."
        )


def _prepare_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Copie propre + colonne repeat_monetary garantie."""
    need = {"CustomerID", "frequency", "recency", "T"}
    missing = need - set(summary.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans summary : {sorted(missing)}")

    df = summary.copy()
    df["frequency"] = df["frequency"].astype(float)
    df.loc[df["frequency"] == 0, "recency"] = 0.0      # coherence BG/NBD

    if "repeat_monetary" not in df.columns:
        warnings.warn(
            "Colonne 'repeat_monetary' absente : 'monetary' (qui inclut le 1er achat) "
            "est utilisee pour Gamma-Gamma. Resultats moins precis.",
            stacklevel=3,
        )
        df["repeat_monetary"] = df["monetary"].where(df["frequency"] > 0, 0.0)
    df["repeat_monetary"] = df["repeat_monetary"].fillna(0.0)

    _check_rfm(df["frequency"].to_numpy(), df["recency"].to_numpy(float),
               df["T"].to_numpy(float))
    return df


def _select_penalizer(name, model_factory, fit_fn, heldout_ll_fn, n, grid, seed):
    """
    Choix du penalisateur par validation croisee K blocs sur la
    log-vraisemblance hors-echantillon. Regle du 1-ecart-type : on prend le
    penalisateur le PLUS GRAND dont le score reste dans 1 ecart-type du
    meilleur -> modele plus stable.
    """
    if n < CV_FOLDS * 20:
        return 0.0, {}
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), CV_FOLDS)
    scores = {}
    for lam in grid:
        per_fold = []
        for k in range(CV_FOLDS):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(CV_FOLDS) if j != k])
            try:
                model = model_factory(lam)
                fit_fn(model, tr)
                per_fold.append(heldout_ll_fn(model, te))
            except Exception:
                per_fold.append(np.nan)
        arr = np.array(per_fold, dtype=float)
        if np.isfinite(arr).all():
            scores[lam] = (arr.mean(), arr.std(ddof=1) / np.sqrt(CV_FOLDS))
    if not scores:
        return 0.0, {}
    best_lam = max(scores, key=lambda k: scores[k][0])
    best_mean, best_se = scores[best_lam]
    eligible = [lam for lam, (mu, _) in scores.items() if mu >= best_mean - best_se]
    chosen = max(eligible)
    print(f"  {name} : penalisateur choisi = {chosen:g} "
          f"(CV {CV_FOLDS} blocs, meilleur brut = {best_lam:g})")
    return chosen, scores


# =============================================================================
# 4. Fonctions publiques (memes noms que l'ancienne version)
# =============================================================================

def fit_bgnbd(summary: pd.DataFrame, tune: bool = True) -> BGNBDModel:
    """Entraine BG/NBD sur tous les clients (repetitifs ou non)."""
    print("\n  Fitting BG/NBD model...")
    df = _prepare_summary(summary)
    x, tx, T = (df[c].to_numpy(float) for c in ("frequency", "recency", "T"))

    lam = 0.0
    if tune:
        lam, _ = _select_penalizer(
            "BG/NBD",
            model_factory=lambda l: BGNBDModel(penalizer=l, n_starts=3),
            fit_fn=lambda m, idx: m.fit(x[idx], tx[idx], T[idx]),
            heldout_ll_fn=lambda m, idx: float(np.mean(
                bgnbd_log_likelihood([m.params_[k] for k in m.PARAM_NAMES],
                                     x[idx], tx[idx], T[idx]))),
            n=len(df), grid=PENALIZER_GRID, seed=RANDOM_STATE,
        )

    bgf = BGNBDModel(penalizer=lam).fit(x, tx, T)
    p = bgf.params_
    print(f"  BG/NBD fitted (converged={bgf.converged_}, n={bgf.n_obs_:,}, LL={bgf.loglik_:,.1f})")
    print(f"  Parameters: r={p['r']:.4f}, alpha={p['alpha']:.4f}, a={p['a']:.4f}, b={p['b']:.4f}")
    print(f"  Mean dropout probability a/(a+b) = {p['a'] / (p['a'] + p['b']):.3f}")
    if bgf.bounds_hit_:
        warnings.warn(f"BG/NBD : parametres colles aux bornes : {bgf.bounds_hit_}. "
                      "Donnees trop pauvres ou peu adaptees au modele.")
    if p["a"] <= 1.0:
        print("  Note : a <= 1 -> l'esperance a long terme est tres sensible ; "
              "preferer des horizons <= 3 ans.")
    return bgf


def predict_purchases(bgf: BGNBDModel, summary: pd.DataFrame,
                      days: float = PREDICTION_DAYS) -> pd.DataFrame:
    """Ajoute predicted_purchases et prob_alive."""
    if days <= 0:
        raise ValueError("days doit etre positif")
    df = summary.copy()
    x, tx, T = (df[c].to_numpy(float) for c in ("frequency", "recency", "T"))
    tx = np.where(x == 0, 0.0, tx)
    df["predicted_purchases"] = bgf.expected_purchases(days, x, tx, T)
    df["prob_alive"] = bgf.prob_alive(x, tx, T)

    print(f"\n  Purchase predictions ({days:g}-day horizon):")
    print(f"    Avg predicted purchases : {df['predicted_purchases'].mean():.3f}")
    print(f"    Avg probability alive   : {df['prob_alive'].mean():.3f}")
    print(f"    Customers likely alive  : {(df['prob_alive'] > 0.5).sum():,} "
          f"({(df['prob_alive'] > 0.5).mean():.1%})")
    return df


def fit_gamma_gamma(summary: pd.DataFrame, tune: bool = True) -> GammaGammaModel:
    """Entraine Gamma-Gamma sur les clients repetitifs (montants hors 1er achat)."""
    print("\n  Fitting Gamma-Gamma model...")
    df = _prepare_summary(summary)
    rep = df[(df["frequency"] > 0) & (df["repeat_monetary"] > 0)]
    print(f"  Using {len(rep):,} repeat buyers for fitting")
    x = rep["frequency"].to_numpy(float)
    m = rep["repeat_monetary"].to_numpy(float)

    # Hypothese cle : frequence et montant moyen independants
    if len(rep) > 30:
        rho = spearmanr(x, m).correlation
        flag = "OK" if abs(rho) < 0.3 else "ATTENTION : hypothese d'independance douteuse"
        print(f"  Spearman(frequency, monetary) = {rho:+.3f}  -> {flag}")

    # le clip est calcule une fois sur l'echantillon complet puis reutilise en CV
    clip = float(np.quantile(m, MONETARY_CLIP_Q)) if MONETARY_CLIP_Q else np.inf
    m_c = np.minimum(m, clip)

    lam = 0.0
    if tune:
        lam, _ = _select_penalizer(
            "Gamma-Gamma",
            model_factory=lambda l: GammaGammaModel(penalizer=l, n_starts=3),
            fit_fn=lambda g, idx: g.fit(x[idx], m_c[idx], clip_quantile=None),
            heldout_ll_fn=lambda g, idx: float(np.mean(
                gamma_gamma_log_likelihood([g.params_[k] for k in g.PARAM_NAMES],
                                           x[idx], m_c[idx]))),
            n=len(rep), grid=PENALIZER_GRID, seed=RANDOM_STATE,
        )

    ggf = GammaGammaModel(penalizer=lam).fit(x, m)
    p = ggf.params_
    print(f"  Gamma-Gamma fitted (converged={ggf.converged_}, n={ggf.n_obs_:,}, "
          f"clip={ggf.clip_value_:,.2f})")
    print(f"  Parameters: p={p['p']:.4f}, q={p['q']:.4f}, v={p['v']:.4f}")
    print(f"  Population mean order value = {p['p'] * p['v'] / (p['q'] - 1):,.2f}")
    if ggf.bounds_hit_:
        warnings.warn(f"Gamma-Gamma : parametres colles aux bornes : {ggf.bounds_hit_}.")
    return ggf


def predict_clv(bgf: BGNBDModel, ggf: GammaGammaModel, summary: pd.DataFrame,
                years: int = PREDICTION_YEARS,
                gross_margin: float = GROSS_MARGIN,
                annual_discount: float = DISCOUNT_RATE) -> pd.DataFrame:
    """
    CLV = sum_mois [ achats attendus du mois x valeur attendue x marge / (1+i)^mois ]
    avec i = taux mensuel equivalent au taux annuel.
    """
    print("\n  Calculating Customer Lifetime Value...")
    df = _prepare_summary(summary)
    x, tx, T = (df[c].to_numpy(float) for c in ("frequency", "recency", "T"))

    df["predicted_avg_value"] = ggf.expected_average_value(x, df["repeat_monetary"].to_numpy(float))

    months = int(years * 12)
    cum = np.column_stack([
        bgf.expected_purchases(mth * DAYS_PER_MONTH, x, tx, T)
        for mth in range(1, months + 1)
    ])
    cum = np.maximum.accumulate(cum, axis=1)               # cumul monotone
    monthly = np.diff(cum, axis=1, prepend=0.0)

    monthly_rate = (1.0 + annual_discount) ** (1.0 / 12.0) - 1.0
    disc = (1.0 + monthly_rate) ** np.arange(1, months + 1)
    df["expected_purchases_horizon"] = cum[:, -1]
    df["bgnbd_clv"] = (monthly / disc).sum(axis=1) * df["predicted_avg_value"].to_numpy() * gross_margin

    if not np.isfinite(df["bgnbd_clv"]).all():
        raise ValueError("CLV : valeurs non finies.")

    df["clv_segment"] = pd.qcut(
        df["bgnbd_clv"].rank(method="first"), q=4,
        labels=["Low Value", "Mid Value", "High Value", "Top Value"],
    )

    print(f"\n  BG/NBD CLV Results ({years}-year horizon):")
    print(f"    Avg CLV per customer : ${df['bgnbd_clv'].mean():,.2f}")
    print(f"    Total portfolio CLV  : ${df['bgnbd_clv'].sum():,.2f}")
    print(f"    Median CLV           : ${df['bgnbd_clv'].median():,.2f}")
    return df


def validate_model(bgf: BGNBDModel, summary_valid: pd.DataFrame,
                   days: float = None, ggf: GammaGammaModel = None):
    """
    Compare predictions et realite sur la periode de validation.
    Calcule aussi des modeles de reference pour montrer le gain reel :
      naive_rate : on extrapole le rythme historique (n_achats / age) sans churn
      constant   : on predit la moyenne pour tout le monde
    """
    print("\n  Validating BG/NBD model...")
    if "actual_future_purchases" not in summary_valid.columns:
        raise ValueError("Colonne actual_future_purchases manquante")

    if days is None:
        if "holdout_days" in summary_valid.columns:
            days = float(summary_valid["holdout_days"].iloc[0])
        else:
            days = float(summary_valid.attrs.get("validation_days", PREDICTION_DAYS))

    df = predict_purchases(bgf, summary_valid, days=days)
    actual = df["actual_future_purchases"].to_numpy(float)
    pred = df["predicted_purchases"].to_numpy(float)

    def _rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    corr = float(np.corrcoef(actual, pred)[0, 1]) if actual.std() > 0 and pred.std() > 0 else np.nan
    rho = float(spearmanr(actual, pred).correlation)

    naive = (df["frequency"] + 1.0) / df["T"].clip(lower=30.0) * days   # rythme historique, sans churn
    metrics = {
        "mae": float(np.mean(np.abs(actual - pred))),
        "rmse": _rmse(actual, pred),
        "correlation": corr,
        "spearman": rho,
        "actual_mean": float(actual.mean()),
        "pred_mean": float(pred.mean()),
        "aggregate_error_pct": float((pred.sum() - actual.sum()) / max(actual.sum(), _EPS) * 100),
        "rmse_naive_rate": _rmse(actual, naive.to_numpy()),
        "rmse_constant": _rmse(actual, np.full_like(actual, actual.mean())),
        "holdout_days": float(days),
    }

    print(f"\n  Model Validation ({days:g}-day prediction):")
    print(f"    MAE                  : {metrics['mae']:.4f} purchases")
    print(f"    RMSE                 : {metrics['rmse']:.4f}  "
          f"(naive rate: {metrics['rmse_naive_rate']:.4f}, constant: {metrics['rmse_constant']:.4f})")
    print(f"    Correlation          : {corr:.4f} (Spearman {rho:.4f})")
    print(f"    Actual / predicted   : {actual.mean():.4f} / {pred.mean():.4f} "
          f"(total error {metrics['aggregate_error_pct']:+.1f}%)")

    # --- Gamma-Gamma et revenu, si les colonnes existent ---
    if ggf is not None and "actual_future_revenue" in df.columns:
        x = df["frequency"].to_numpy(float)
        m = df["repeat_monetary"].fillna(0.0).to_numpy(float) if "repeat_monetary" in df else np.zeros(len(df))
        df["predicted_avg_value"] = ggf.expected_average_value(x, m)
        df["predicted_revenue"] = df["predicted_purchases"] * df["predicted_avg_value"]

        rev = df["actual_future_revenue"].to_numpy(float)
        prev = df["predicted_revenue"].to_numpy(float)
        metrics["revenue_aggregate_error_pct"] = float((prev.sum() - rev.sum()) / max(rev.sum(), _EPS) * 100)
        metrics["revenue_correlation"] = float(np.corrcoef(rev, prev)[0, 1])

        buyers = (df["actual_future_purchases"] > 0) & (x > 0) & (m > 0)
        if buyers.sum() >= 10:
            act_val = (rev[buyers] / df.loc[buyers, "actual_future_purchases"].to_numpy(float))
            gg_val = df.loc[buyers, "predicted_avg_value"].to_numpy(float)
            raw_val = np.minimum(m[buyers], ggf.clip_value_)
            metrics["gg_mae"] = float(np.mean(np.abs(act_val - gg_val)))
            metrics["gg_mae_naive"] = float(np.mean(np.abs(act_val - raw_val)))
            metrics["gg_mean_actual"] = float(act_val.mean())
            metrics["gg_mean_pred"] = float(gg_val.mean())
            print(f"\n  Gamma-Gamma validation ({int(buyers.sum()):,} repeat buyers who bought again):")
            print(f"    MAE panier moyen     : {metrics['gg_mae']:.2f}  "
                  f"(naive = ancien panier : {metrics['gg_mae_naive']:.2f})")
            print(f"    Panier reel / predit : {metrics['gg_mean_actual']:.2f} / {metrics['gg_mean_pred']:.2f}")
        print(f"    Revenu total : erreur {metrics['revenue_aggregate_error_pct']:+.1f}%, "
              f"correlation {metrics['revenue_correlation']:.3f}")

    df["actual_future_purchases_val"] = actual
    return metrics, df


def summarize_bgnbd_clv(df: pd.DataFrame) -> pd.DataFrame:
    """Resume par segment CLV."""
    seg = (
        df.groupby("clv_segment", observed=True)
        .agg(
            customers=("CustomerID", "count"),
            avg_clv=("bgnbd_clv", "mean"),
            total_clv=("bgnbd_clv", "sum"),
            avg_prob_alive=("prob_alive", "mean"),
            avg_pred_purchases=("predicted_purchases", "mean"),
        )
        .reset_index()
    )
    print("\n  BG/NBD CLV by Segment:")
    print(f"  {'Segment':<15} {'Customers':>10} {'Avg CLV':>12} {'Total CLV':>14} {'P(Alive)':>10}")
    print("  " + "-" * 65)
    for _, row in seg.iterrows():
        print(f"  {row['clv_segment']:<15} {int(row['customers']):>10,} "
              f"${row['avg_clv']:>11,.2f} ${row['total_clv']:>13,.2f} "
              f"{row['avg_prob_alive']:>9.1%}")
    return seg


# =============================================================================
# 5. Pipeline complet
# =============================================================================

def run_bgnbd_pipeline(summary: pd.DataFrame, summary_valid: pd.DataFrame) -> tuple:
    """
    Returns
    -------
    df_clv, segment_summary, bgf, ggf, val_metrics, val_df
    """
    print("\n" + "=" * 50)
    print("  BG/NBD + GAMMA-GAMMA CLV MODEL")
    print("=" * 50)
    print(f"  Gross margin     : {GROSS_MARGIN:.0%}")
    print(f"  Discount rate    : {DISCOUNT_RATE:.0%} / an")
    print(f"  Prediction years : {PREDICTION_YEARS}")

    bgf = fit_bgnbd(summary)
    df_tmp = predict_purchases(bgf, _prepare_summary(summary))
    ggf = fit_gamma_gamma(summary)
    df_clv = predict_clv(bgf, ggf, df_tmp)
    segment_summary = summarize_bgnbd_clv(df_clv)
    val_metrics, val_df = validate_model(bgf, summary_valid, ggf=ggf)
    return df_clv, segment_summary, bgf, ggf, val_metrics, val_df
