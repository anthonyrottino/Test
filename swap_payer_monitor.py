"""
G4 10y Payer Monitor — USD SOFR / EUR €STR / GBP SONIA / CAD CORRA.
Single-file Streamlit app.

Per currency: outright momentum, 2y policy repricing, 10s30s (term premium),
breakevens, swap spread vs govie, carry+roll per unit of vol for a 10y payer.
Cross vs USD: X–USD 10y spread, rolling beta, beta-adjusted residual.
EUR extras: Bund ASW (6m Euribor), OAT/Bund and BTP/Bund as a France /
flight-to-quality flag. Everything rolls into a transparent scorecard + alerts.

Run:   streamlit run swap_payer_monitor.py
Data:  Bloomberg via xbbg (needs Terminal) or synthetic "Demo" mode.
Verify every ticker below on your Terminal before relying on the output.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
# s2/s9/s10/s30: OIS swaps · gov10: 10y govie · be10: 10y inflation swap
# (None = not used) · fix: overnight fixing
CCYS: dict[str, dict] = {
    "usd": dict(pfx="USOSFR", name="USD SOFR", s2="USOSFR2 Curncy", s9="USOSFR9 Curncy",
                s10="USOSFR10 Curncy", s30="USOSFR30 Curncy",
                gov10="USGG10YR Index", be10="USSWIT10 Curncy", fix="SOFRRATE Index"),
    "eur": dict(pfx="EESWE", name="EUR €STR", s2="EESWE2 Curncy", s9="EESWE9 Curncy",
                s10="EESWE10 Curncy", s30="EESWE30 Curncy",
                gov10="GDBR10 Index", be10="EUSWI10 Curncy", fix="ESTRON Index"),
    "gbp": dict(pfx="BPSWS", name="GBP SONIA", s2="BPSWS2 Curncy", s9="BPSWS9 Curncy",
                s10="BPSWS10 Curncy", s30="BPSWS30 Curncy",
                gov10="GUKG10 Index", be10="BPSWIT10 Curncy", fix="SONIO/N Index"),
    "cad": dict(pfx="CDSO", name="CAD CORRA", s2="CDSO2 Curncy", s9="CDSO9 Curncy",
                s10="CDSO10 Curncy", s30="CDSO30 Curncy",
                gov10="GCAN10YR Index", be10=None, fix="CAONREPO Index"),
}
EXTRA = {
    "eur6m_10y": "EUSA10 Curncy",   # EUR 6m Euribor 10y (Bund ASW)
    "oat_10y": "GFRN10 Index",
    "btp_10y": "GBTPGR10 Index",
    "move": "MOVE Index",
}
FIELDS = ["s2", "s9", "s10", "s30", "gov10", "be10", "fix"]
TENORS = [2, 5, 10, 15, 20, 25, 30]   # curve-RV PCA tenors: f"{pfx}{t} Curncy"
CROSSES = [c for c in CCYS if c != "usd"]

Z_WIN = 252     # valuation z-score lookback (days)
BETA_WIN = 60   # rolling beta window
MOM_WIN = 20    # momentum horizon
VOL_WIN = 63    # realized vol window
CLIP = 2.0      # cap on each signal contribution
BP = 100.0
PCA_WIN = 252   # PCA fit window (days)
N_PC = 3        # level / slope / curvature
PCA_OOS = 20    # out-of-sample days: fit ends this many days before today
PCA_HL = 84     # EW half-life for mean / vol / correlation (days, 0 = equal-weighted)

ALERTS = {
    "oat_bund_z": 1.5,
    "oat_bund_20d": 15.0,      # bp
    "move_20d": 15.0,          # index pts
    "curve_20d": 10.0,         # 10s30s, bp
    "swap_spread_20d": 5.0,    # bp
    "resid_mom_z": 2.0,
    "beta_shift": 0.20,
}


def ticker_map() -> dict[str, str]:
    m = {f"{c}_{k}": cfg[k] for c, cfg in CCYS.items() for k in FIELDS if cfg.get(k)}
    for c, cfg in CCYS.items():
        for t in TENORS:
            m[f"{c}_c{t}"] = f"{cfg['pfx']}{t} Curncy"
    m.update(EXTRA)
    return m


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
def bloomberg_available() -> bool:
    """True when xbbg can be imported and APP_MODE is not forced to 'demo'
    (set APP_MODE=demo, e.g. in Streamlit Cloud secrets, to hide Bloomberg)."""
    import importlib.util
    import os
    if os.environ.get("APP_MODE", "").lower() == "demo":
        return False
    return importlib.util.find_spec("xbbg") is not None


def fetch_bloomberg(start: dt.date, end: dt.date) -> tuple[pd.DataFrame, list[str]]:
    from xbbg import blp

    tm = ticker_map()
    raw = blp.bdh(sorted(set(tm.values())), "PX_LAST", start, end)
    if raw is None or raw.empty:
        raise RuntimeError("Bloomberg returned no data — check Terminal connection.")
    raw.columns = raw.columns.get_level_values(0)
    px = pd.DataFrame({k: raw[t] if t in raw.columns else np.nan for k, t in tm.items()},
                      index=pd.to_datetime(raw.index))
    missing = sorted({t for t in tm.values() if t not in raw.columns})
    core = [CCYS[c]["s10"] for c in CCYS]
    if any(t in missing for t in core):
        raise RuntimeError(f"Missing core 10y tickers: {[t for t in missing if t in core]}")
    return px.sort_index().ffill().dropna(how="all"), missing


# --------------------------------------------------------------------------
# PUBLIC DATA: government bond yields + overnight fixings (free, no licence)
# --------------------------------------------------------------------------
# Government curves stand in for the OIS curves; fixings are the real RFRs.
# Swap spreads, France stress, Bund ASW and MOVE are not available (skipped).
PUBLIC = {
    "usd": dict(src="fred", curve={2: "DGS2", 5: "DGS5", 7: "DGS7", 10: "DGS10", 20: "DGS20", 30: "DGS30"},
                fix="SOFR", be10="T10YIE"),
    "eur": dict(src="ecb", curve={t: f"SR_{t}Y" for t in (2, 5, 9, 10, 15, 20, 25, 30)},
                fix="EST/B.EU000A2X2A25.WT"),
    "gbp": dict(src="boe", curve={5: "IUDMNZC", 10: "IUDSNZC", 20: "IUDLNZC"}, fix="IUDSOIA"),
    "cad": dict(src="boc", curve={2: "BD.CDN.2YR.DQ.YLD", 5: "BD.CDN.5YR.DQ.YLD", 7: "BD.CDN.7YR.DQ.YLD",
                                  10: "BD.CDN.10YR.DQ.YLD", 30: "BD.CDN.LONG.DQ.YLD"}, fix="AVG.INTWO"),
}
PUBLIC_LABEL = {"usd": "US Treasuries (FRED)", "eur": "Euro area AAA curve (ECB)",
                "gbp": "Gilts, nominal zero-coupon (BoE)", "cad": "GoC benchmarks (Bank of Canada)"}
_UA = {"User-Agent": "Mozilla/5.0 (G4 payer monitor; research use)"}


def _get(url: str, timeout: int = 40) -> str:
    import requests
    r = requests.get(url, headers=_UA, timeout=timeout)
    r.raise_for_status()
    return r.text


def parse_fred_csv(text: str) -> pd.DataFrame:
    import io
    df = pd.read_csv(io.StringIO(text))
    df.index = pd.to_datetime(df.iloc[:, 0])
    return df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")


def parse_ecb_csv(text: str, key_col: str) -> pd.Series | pd.DataFrame:
    import io
    df = pd.read_csv(io.StringIO(text))
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    if key_col in df:
        return df.pivot_table(index="TIME_PERIOD", columns=key_col, values="OBS_VALUE")
    return df.set_index("TIME_PERIOD")["OBS_VALUE"]


def parse_boc_json(text: str) -> pd.DataFrame:
    import json
    obs = json.loads(text)["observations"]
    rows = {o["d"]: {k: v.get("v") for k, v in o.items() if isinstance(v, dict)} for o in obs}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(df.index)
    return df.apply(pd.to_numeric, errors="coerce")


def parse_boe_csv(text: str) -> pd.DataFrame:
    import io
    df = pd.read_csv(io.StringIO(text))
    df.index = pd.to_datetime(df.iloc[:, 0], format="%d %b %Y", errors="coerce")
    return df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")[lambda x: x.index.notna()]


def _fetch_source(c: str, cfg: dict, start: dt.date) -> pd.DataFrame:
    """One DataFrame per currency: columns = curve tenors (int) + 'fix' (+ 'be10')."""
    codes = list(cfg["curve"].values())
    if cfg["src"] == "fred":
        ids = codes + [cfg["fix"]] + ([cfg["be10"]] if cfg.get("be10") else [])
        raw = parse_fred_csv(_get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(ids)
                                  + f"&cosd={start:%Y-%m-%d}"))
        out = pd.DataFrame({t: raw[code] for t, code in cfg["curve"].items()})
        out["fix"] = raw[cfg["fix"]]
        if cfg.get("be10"):
            out["be10"] = raw[cfg["be10"]]
    elif cfg["src"] == "ecb":
        base = "https://data-api.ecb.europa.eu/service/data/"
        yc = parse_ecb_csv(_get(f"{base}YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{'+'.join(codes)}"
                                f"?format=csvdata&startPeriod={start:%Y-%m-%d}"), "DATA_TYPE_FM")
        out = pd.DataFrame({t: yc[code] for t, code in cfg["curve"].items() if code in yc})
        fx = parse_ecb_csv(_get(f"{base}{cfg['fix']}?format=csvdata&startPeriod={start:%Y-%m-%d}"), "_none_")
        out["fix"] = fx
    elif cfg["src"] == "boe":
        url = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes"
               f"&Datefrom={start:%d/%b/%Y}&Dateto=now&SeriesCodes={','.join(codes + [cfg['fix']])}"
               "&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
        raw = parse_boe_csv(_get(url))
        out = pd.DataFrame({t: raw[code] for t, code in cfg["curve"].items() if code in raw})
        out["fix"] = raw[cfg["fix"]] if cfg["fix"] in raw else np.nan
    elif cfg["src"] == "boc":
        ids = codes + [cfg["fix"]]
        raw = parse_boc_json(_get("https://www.bankofcanada.ca/valet/observations/" + ",".join(ids)
                                  + f"/json?start_date={start:%Y-%m-%d}"))
        out = pd.DataFrame({t: raw[code] for t, code in cfg["curve"].items() if code in raw})
        out["fix"] = raw[cfg["fix"]] if cfg["fix"] in raw else np.nan
    else:
        raise ValueError(cfg["src"])
    return out.sort_index()


def assemble_public(frames: dict[str, pd.DataFrame], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Map per-currency government curves onto the model's columns (same layout as Bloomberg)."""
    idx = pd.bdate_range(start, end)
    cols = {k: np.nan for k in ticker_map()}
    px = pd.DataFrame(cols, index=idx)

    def interp(d: pd.DataFrame, t: float) -> pd.Series:
        ts = sorted(c for c in d.columns if isinstance(c, (int, np.integer)))
        if t in ts:
            return d[t]
        lo = max([x for x in ts if x < t], default=None)
        hi = min([x for x in ts if x > t], default=None)
        if lo is None or hi is None:
            return pd.Series(np.nan, index=d.index)
        w = (t - lo) / (hi - lo)
        return (1 - w) * d[lo] + w * d[hi]

    for c, d in frames.items():
        d = d.reindex(idx).ffill(limit=5)
        for t in TENORS:
            if t in d.columns:
                px[f"{c}_c{t}"] = d[t]
        for k, t in [("s2", 2), ("s9", 9), ("s10", 10), ("s30", 30)]:
            px[f"{c}_{k}"] = interp(d, t)
        px[f"{c}_fix"] = d["fix"] if "fix" in d else np.nan
        if "be10" in d:
            px[f"{c}_be10"] = d["be10"]
    return px.dropna(subset=[f"{c}_s10" for c in CCYS], how="any")


def fetch_public(start: dt.date, end: dt.date) -> tuple[pd.DataFrame, list[str]]:
    """Free government-bond and fixing data from FRED, ECB, BoE and Bank of Canada."""
    frames, problems = {}, []
    for c, cfg in PUBLIC.items():
        try:
            frames[c] = _fetch_source(c, cfg, start)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{PUBLIC_LABEL[c]}: {type(e).__name__}: {e}"[:200])
    missing_core = [c for c in CCYS if c not in frames]
    if missing_core:
        raise RuntimeError("Public data unavailable for " + ", ".join(c.upper() for c in missing_core)
                           + " — " + " | ".join(problems))
    px = assemble_public(frames, start, end)
    if px.empty:
        raise RuntimeError("Public data loaded but no common dates with all four 10y yields.")
    notes = problems + ["Government bond yields used as OIS proxies; swap spreads, France stress, "
                        "Bund ASW, MOVE and non-USD breakevens are not available from public sources."]
    return px, notes


def make_demo(start: dt.date, end: dt.date, seed: int = 7) -> tuple[pd.DataFrame, list[str]]:
    """Synthetic, mean-reverting, structurally sensible data (incl. a France shock)."""
    idx = pd.bdate_range(start, end)
    n = len(idx)
    r = np.random.default_rng(seed)
    N = lambda s: r.normal(0, s, n)

    def ou(x, k=0.015):
        out, acc = np.empty(n), 0.0
        for i, v in enumerate(x):
            acc = acc * (1 - k) + v
            out[i] = acc
        return out / BP  # bp -> %

    g = N(3.5)                       # global rates factor
    fr = N(1.0)
    k = int(n * 0.85)
    fr[k:k + 12] += 7.0
    fr[k + 12:k + 25] -= 3.0

    # lvl, beta to global, local vol, fixing, 10s30s, swap-govie, breakeven
    p = {
        "usd": (4.20, 1.0, 2.5, 3.90, 0.30, -0.25, 2.40),
        "eur": (2.60, 0.6, 2.0, 2.40, 0.15, -0.05, 2.10),
        "gbp": (4.10, 0.8, 3.0, 4.00, 0.25, -0.40, 3.30),
        "cad": (3.20, 0.8, 2.0, 2.75, 0.15, -0.10, None),
    }
    # recent currency-specific trends (bp/day over the last 45 days) so the
    # demo shows both payer and receiver set-ups
    trend = {"usd": 2.2, "eur": 2.0, "gbp": -1.2, "cad": -0.9}
    tr_mask = np.zeros(n)
    tr_mask[-25:] = 1.0
    d = {}
    for c, (lvl, b, lv, fix, slope, ss, be) in p.items():
        loc = N(lv) + trend[c] * tr_mask
        shock = b * g + loc - (0.4 * fr if c == "eur" else 0)
        s10 = lvl + ou(shock)
        d[f"{c}_s10"] = s10
        d[f"{c}_s2"] = (lvl - 0.30) + ou(0.6 * b * g + 1.3 * loc + N(1.5))
        d[f"{c}_s9"] = s10 - 0.04 + ou(N(0.25))
        d[f"{c}_s30"] = s10 + slope + ou(N(0.9) + 0.4 * trend[c] * tr_mask)
        gov_extra = -0.5 * fr if c == "eur" else 0
        d[f"{c}_gov10"] = s10 - ss + ou(N(0.6) + gov_extra)
        d[f"{c}_be10"] = (be + ou(N(1.1))) if be is not None else np.full(n, np.nan)
        d[f"{c}_fix"] = np.full(n, fix)
        # full curve: piecewise-linear through 2/10/30 + hump + idiosyncratic OU
        for t in TENORS:
            if t in (2, 10, 30):
                d[f"{c}_c{t}"] = d[f"{c}_s{t}"]
                continue
            lo, hi = (2, 10) if t < 10 else (10, 30)
            wgt = (t - lo) / (hi - lo)
            base = (1 - wgt) * d[f"{c}_s{lo}"] + wgt * d[f"{c}_s{hi}"]
            hump = 0.04 * np.sin(np.pi * (t - 2) / 28)
            d[f"{c}_c{t}"] = base + hump + ou(N(0.6), k=0.05)
    d["eur6m_10y"] = d["eur_s10"] + 0.12 + ou(N(0.3))
    d["oat_10y"] = d["eur_gov10"] + 0.75 + ou(fr)
    d["btp_10y"] = d["eur_gov10"] + 0.90 + ou(0.5 * fr + N(1.5))
    d["move"] = np.clip(100 + np.cumsum(N(1.5)), 50, None)
    return pd.DataFrame(d, index=idx), []


# --------------------------------------------------------------------------
# ANALYTICS
# --------------------------------------------------------------------------
def zscore(s: pd.Series, win: int = Z_WIN) -> pd.Series:
    m = s.rolling(win, min_periods=win // 2).mean()
    sd = s.rolling(win, min_periods=win // 2).std()
    return (s - m) / sd


def mom_z(s_bp: pd.Series, win: int = MOM_WIN) -> pd.Series:
    """Vol-adjusted momentum: win-day change / (daily vol * sqrt(win))."""
    vol = s_bp.diff().rolling(VOL_WIN, min_periods=20).std()
    return s_bp.diff(win) / (vol * np.sqrt(win))


def build_features(px: pd.DataFrame) -> pd.DataFrame:
    import warnings
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    f = pd.DataFrame(index=px.index)
    for c in CCYS:
        s10, s9 = px[f"{c}_s10"], px[f"{c}_s9"]
        f[f"{c}_2y"] = px[f"{c}_s2"]
        f[f"{c}_10y"] = s10
        f[f"{c}_10s30s"] = (px[f"{c}_s30"] - s10) * BP
        f[f"{c}_swap_spread"] = (s10 - px[f"{c}_gov10"]) * BP
        f[f"{c}_be"] = px[f"{c}_be10"] if f"{c}_be10" in px else np.nan
        vol = (s10.diff() * BP).rolling(VOL_WIN, min_periods=20).std() * np.sqrt(252)
        carry = (px[f"{c}_fix"] - s10) * BP * 0.25   # payer receives float
        roll = -(s10 - s9) * BP * 0.25                # rolls down to 9.75y
        f[f"{c}_vol"] = vol
        f[f"{c}_cr3m"] = carry + roll
        f[f"{c}_cr_vol"] = f[f"{c}_cr3m"] / (vol * np.sqrt(0.25))

        # every curve tenor: level, vol, carry + roll (roll from local curve slope)
        avail = [t for t in TENORS if f"{c}_c{t}" in px and px[f"{c}_c{t}"].notna().any()]
        for i, t in enumerate(avail):
            st_ = px[f"{c}_c{t}"]
            f[f"{c}_t{t}"] = st_
            if t == 10:
                f[f"{c}_t10_vol"], f[f"{c}_t10_cr3m"] = vol, f[f"{c}_cr3m"]
            else:
                if i == 0:
                    nb = avail[1]
                    slope = (px[f"{c}_c{nb}"] - st_) / (nb - t)
                else:
                    pv = avail[i - 1]
                    slope = (st_ - px[f"{c}_c{pv}"]) / (t - pv)
                f[f"{c}_t{t}_vol"] = (st_.diff() * BP).rolling(VOL_WIN, min_periods=20).std() * np.sqrt(252)
                f[f"{c}_t{t}_cr3m"] = (px[f"{c}_fix"] - st_) * BP * 0.25 - slope * BP * 0.25
            f[f"{c}_t{t}_cr_vol"] = f[f"{c}_t{t}_cr3m"] / (f[f"{c}_t{t}_vol"] * np.sqrt(0.25))

    d_usd = px.usd_s10.diff() * BP
    for c in CROSSES:
        d_x = px[f"{c}_s10"].diff() * BP
        beta = d_x.rolling(BETA_WIN).cov(d_usd) / d_usd.rolling(BETA_WIN).var()
        f[f"{c}_usd_10y"] = (px[f"{c}_s10"] - px.usd_s10) * BP
        f[f"{c}_usd_2y"] = (px[f"{c}_s2"] - px.usd_s2) * BP
        f[f"{c}_beta"] = beta
        f[f"{c}_beta_lt"] = beta.rolling(250, min_periods=60).mean()
        f[f"{c}_resid"] = (d_x - beta.shift(1) * d_usd).fillna(0).cumsum()

    f["eur_bund_asw"] = (px.eur6m_10y - px.eur_gov10) * BP
    f["oat_bund"] = (px.oat_10y - px.eur_gov10) * BP
    f["btp_bund"] = (px.btp_10y - px.eur_gov10) * BP
    f["move"] = px.move
    return f.copy()


def _last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if s.size else np.nan


def _clip(x: float) -> float:
    return 0.0 if x is None or not np.isfinite(x) else float(np.clip(x, -CLIP, CLIP))


def signal_history(f: pd.DataFrame) -> dict[str, tuple[dict[str, float], pd.DataFrame]]:
    """Full time series of every raw signal, per trade, with its hand weight.
    Positive weight x positive raw = supportive. Used by the scorecard (last row)
    and by the walk-forward backtest (whole history)."""
    fr_pen = -zscore(f.oat_bund).clip(lower=0)

    def outright(c: str):
        sig = {
            "10y momentum (vol-adj)": (1.0, mom_z(f[f"{c}_10y"] * BP)),
            "2y policy repricing": (1.0, mom_z(f[f"{c}_2y"] * BP)),
            "10s30s steepening (term premium)": (0.5, mom_z(f[f"{c}_10s30s"])),
        }
        if f[f"{c}_be"].notna().any():
            sig["Breakeven momentum"] = (0.5, mom_z(f[f"{c}_be"] * BP))
        sig["Carry+roll / vol (3m)"] = (1.0, f[f"{c}_cr_vol"])
        sig["Valuation (1y z, mean-rev)"] = (-0.5, zscore(f[f"{c}_10y"]))
        if c == "eur":
            sig["France stress (flight to Bunds)"] = (1.0, fr_pen)
        return sig

    def cross(c: str):
        sig = {
            "Beta-adj residual momentum": (1.0, mom_z(f[f"{c}_resid"])),
            "2y differential momentum": (1.0, mom_z(f[f"{c}_usd_2y"])),
            "Relative carry+roll / vol": (1.0, f[f"{c}_cr_vol"] - f.usd_cr_vol),
            "Spread valuation (1y z)": (-0.5, zscore(f[f"{c}_usd_10y"])),
        }
        if c == "eur":
            sig["France stress"] = (1.0, fr_pen)
        return sig

    def steepener(c: str):
        return {
            "10s30s momentum": (1.0, mom_z(f[f"{c}_10s30s"])),
            "10s30s valuation (1y z)": (-0.5, zscore(f[f"{c}_10s30s"])),
            "Swap spread momentum (govie supply)": (-0.5, mom_z(f[f"{c}_swap_spread"])),
        }

    trades: dict[str, dict] = {}
    for c in CCYS:
        trades[f"Pay {CCYS[c]['name']} 10y"] = outright(c)
    for c in CROSSES:
        trades[f"Pay {c.upper()} / Rec USD 10y (beta-adj)"] = cross(c)
    for c in ["eur", "gbp"]:
        trades[f"{c.upper()} 10s30s steepener"] = steepener(c)

    return {name: ({k: w for k, (w, _) in sig.items()},
                   pd.DataFrame({k: v for k, (_, v) in sig.items()}, index=f.index))
            for name, sig in trades.items()}


def scorecard(f: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Signal contributions per trade, latest date. Positive = supportive."""
    out = {}
    for name, (w, S) in signal_history(f).items():
        rows = [(k, w[k], _last(S[k])) for k in S.columns]
        df = pd.DataFrame(rows, columns=["Signal", "Weight", "Raw"])
        df["Contribution"] = [_clip(wt * r) for wt, r in zip(df.Weight, df.Raw)]
        out[name] = df
    return out


def tenor_signals(f: pd.DataFrame, c: str, t: int) -> dict[str, tuple[float, pd.Series]]:
    """Payer signals for the t-year swap of currency c (positive = pay, negative = receive)."""
    sig = {
        "Rate momentum (vol-adj)": (1.0, mom_z(f[f"{c}_t{t}"] * BP)),
        "2y policy repricing": (1.0, mom_z(f[f"{c}_2y"] * BP)),
        # term premium matters more the longer the tenor
        "10s30s steepening (term premium)": (round(0.5 * min(t / 10, 2.0), 2), mom_z(f[f"{c}_10s30s"])),
    }
    if f[f"{c}_be"].notna().any():
        sig["Breakeven momentum"] = (0.5, mom_z(f[f"{c}_be"] * BP))
    sig["Carry+roll / vol (3m)"] = (1.0, f[f"{c}_t{t}_cr_vol"])
    sig["Valuation (1y z, mean-rev)"] = (-0.5, zscore(f[f"{c}_t{t}"]))
    if c == "eur":
        sig["France stress (flight to Bunds)"] = (1.0, -zscore(f.oat_bund).clip(lower=0))
    return sig


def side_of(score: float) -> str:
    return "Pay" if score >= CONVICTION else "Receive" if score <= -CONVICTION else "Neutral"


def directional_matrix(f: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Score of every (currency, tenor) outright. > 0 pay, < 0 receive."""
    rows, detail = [], {}
    for c in CCYS:
        for t in TENORS:
            if f"{c}_t{t}" not in f:
                continue
            sig = tenor_signals(f, c, t)
            df = pd.DataFrame([(k, w, _last(v)) for k, (w, v) in sig.items()],
                              columns=["Signal", "Weight", "Raw"])
            df["Contribution"] = [_clip(w * r) for w, r in zip(df.Weight, df.Raw)]
            sc_ = float(df.Contribution.sum())
            detail[(c, t)] = df
            rows.append(dict(ccy=c, Currency=CCYS[c]["name"], Tenor=t,
                             Rate=_last(f[f"{c}_t{t}"]), Score=sc_, Side=side_of(sc_),
                             CR3m=_last(f[f"{c}_t{t}_cr3m"]), Vol=_last(f[f"{c}_t{t}_vol"])))
    return pd.DataFrame(rows), detail


def verdict(total: float) -> str:
    return "Supportive" if total >= 1.5 else "Against" if total <= -1.5 else "Neutral"


def strategy_label(name: str, score: float) -> str:
    """Trade name in its preferred direction: payer or receiver, steepener or flattener."""
    kind, c = _trade_meta(name)
    up = score >= 0
    if kind == "outright":
        return f"{'Pay' if up else 'Receive'} {CCYS[c]['name']} 10y"
    if kind == "cross":
        return (f"Pay {c.upper()} / Rec USD 10y (beta-adj)" if up
                else f"Rec {c.upper()} / Pay USD 10y (beta-adj)")
    if kind == "steepener":
        return f"{c.upper()} 10s30s {'steepener' if up else 'flattener'}"
    return name


def neutral_name(name: str) -> str:
    """Direction-free name (the backtest trades both sides)."""
    kind, c = _trade_meta(name)
    if kind == "outright":
        return f"{CCYS[c]['name']} 10y"
    if kind == "cross":
        return f"{c.upper()} vs USD 10y (beta-adj)"
    if kind == "steepener":
        return f"{c.upper()} 10s30s"
    return name


def strength(conv: float) -> str:
    return "Strong" if conv >= 2.5 else "Moderate" if conv >= CONVICTION else "Weak"


def summary_table(sc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every strategy in its preferred direction (pay or receive), ranked by conviction."""
    rows = []
    for k, v in sc.items():
        score = float(v.Contribution.sum())
        rows.append(dict(Trade=k, Strategy=strategy_label(k, score), Score=score,
                         Conviction=abs(score), Read=strength(abs(score))))
    return pd.DataFrame(rows).sort_values("Conviction", ascending=False).reset_index(drop=True)


def oriented_detail(df: pd.DataFrame, score: float) -> pd.DataFrame:
    """Contributions read from the side shown (positive = supports that side)."""
    sgn = 1.0 if score >= 0 else -1.0
    return df.assign(Weight=df.Weight * sgn, Contribution=df.Contribution * sgn)


def alerts(f: pd.DataFrame) -> list[str]:
    a: list[str] = []
    chg = lambda col: f[col].diff(MOM_WIN).iloc[-1]

    oz = _last(zscore(f.oat_bund))
    if (np.isfinite(oz) and oz > ALERTS["oat_bund_z"]) or chg("oat_bund") > ALERTS["oat_bund_20d"]:
        a.append(f"🇫🇷 France stress: OAT/Bund {f.oat_bund.iloc[-1]:.0f}bp (z {oz:.1f}, "
                 f"20d {chg('oat_bund'):+.0f}bp) — Bund richening caps EUR payers.")
    if chg("move") > ALERTS["move_20d"]:
        a.append(f"📈 MOVE +{chg('move'):.0f} over 20d — term premium bid, size down.")
    for c in CCYS:
        nm = CCYS[c]["name"]
        if abs(chg(f"{c}_10s30s")) > ALERTS["curve_20d"]:
            a.append(f"📐 {nm} 10s30s {chg(f'{c}_10s30s'):+.0f}bp over 20d — long-end flow active.")
        if abs(chg(f"{c}_swap_spread")) > ALERTS["swap_spread_20d"]:
            a.append(f"🏦 {nm} 10y swap spread {chg(f'{c}_swap_spread'):+.1f}bp over 20d.")
    for c in CROSSES:
        rz = _last(mom_z(f[f"{c}_resid"]))
        if np.isfinite(rz) and abs(rz) > ALERTS["resid_mom_z"]:
            a.append(f"⚖️ {c.upper()} vs USD beta-adj residual momentum z {rz:+.1f}.")
        b, blt = _last(f[f"{c}_beta"]), _last(f[f"{c}_beta_lt"])
        if np.isfinite(b) and np.isfinite(blt) and abs(b - blt) > ALERTS["beta_shift"]:
            a.append(f"🔀 {c.upper()}/USD 10y beta {b:.2f} vs 1y avg {blt:.2f} — hedge ratio shifting.")
    return a


def curve_frame(px: pd.DataFrame, c: str) -> pd.DataFrame:
    cols = {t: px[f"{c}_c{t}"] for t in TENORS if f"{c}_c{t}" in px}
    cv = pd.DataFrame(cols)
    return cv.loc[:, cv.notna().mean() > 0.9].dropna()


def half_life(x: pd.Series) -> float:
    """OU half-life (days) from AR(1) on the series."""
    x = x.dropna()
    if len(x) < 30:
        return np.nan
    y, xl = x.diff().iloc[1:].values, x.shift(1).iloc[1:].values
    xl = xl - xl.mean()
    b = np.dot(xl, y - y.mean()) / np.dot(xl, xl)
    return float(-np.log(2) / np.log(1 + b)) if -1 < b < 0 else np.nan


def ew_weights(n: int, hl: float) -> np.ndarray:
    """Exponential weights, newest obs = 1; hl <= 0 -> equal weights."""
    if not hl or hl <= 0:
        return np.ones(n)
    return 0.5 ** ((n - 1 - np.arange(n)) / hl)


def weighted_moments(x: pd.DataFrame, w: np.ndarray):
    """Weighted mean, std and correlation matrix of the columns of x."""
    wn = w / w.sum()
    v = x.values
    mu = wn @ v
    dev = v - mu
    cov = (wn[:, None] * dev).T @ dev
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    return (pd.Series(mu, index=x.columns), pd.Series(sd, index=x.columns), corr)


def sorted_eig(corr: np.ndarray, tenors) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition sorted desc with stable sign conventions:
    PC1 parallel up, PC2 steepener, PC3 belly up."""
    eigval, eigvec = np.linalg.eigh(corr)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    t = np.asarray(tenors, dtype=float)
    if eigvec[:, 0].sum() < 0:
        eigvec[:, 0] *= -1
    if eigvec.shape[1] > 1 and eigvec[-1, 1] - eigvec[0, 1] < 0:
        eigvec[:, 1] *= -1
    if eigvec.shape[1] > 2:
        belly = (t > t.min()) & (t < t.max())
        if eigvec[belly, 2].mean() < 0:
            eigvec[:, 2] *= -1
    return eigval, eigvec


def rolling_loadings(curve: pd.DataFrame, win: int = PCA_WIN, hl: float = PCA_HL,
                     n_pc: int = 3, step: int = 5, n_points: int = 60) -> dict[str, pd.DataFrame]:
    """Eigenvector loadings (unit-norm, corr-PCA) re-estimated every `step` days."""
    dx = (curve * BP).diff().dropna()
    ends = range(len(dx), win - 1, -step)
    ends = sorted(list(ends)[:n_points])
    names = ["Level", "Slope", "Curvature", "PC4"][:n_pc]
    out = {nm: [] for nm in names}
    idx = []
    for e in ends:
        x = dx.iloc[e - win:e]
        _, _, corr = weighted_moments(x, ew_weights(len(x), hl))
        _, vec = sorted_eig(corr, curve.columns)
        for i, nm in enumerate(names):
            out[nm].append(vec[:, i])
        idx.append(dx.index[e - 1])
    return {nm: pd.DataFrame(v, index=idx, columns=[f"{t}y" for t in curve.columns])
            for nm, v in out.items()}


def pca_rv(curve: pd.DataFrame, win: int = PCA_WIN, n_pc: int = N_PC,
           oos: int = PCA_OOS, hl: float = PCA_HL) -> dict:
    """PCA on normalised daily changes ((dx - mean) / std); residual z-scores.

    mu, sd and loadings are fitted on `win` days ending `oos` days ago, then
    applied unchanged through today. (With an in-sample demeaned fit the
    cumulative residual is forced to 0 at the window end, so oos must be > 0.)
    Residual = normalised change minus the part explained by the first n_pc
    PCs, back in bp, cumulated from the start of the fit window.
    hl > 0: mean, vol and correlation (hence loadings / hedge ratios) are
    exponentially weighted; the z-score baseline stays equal-weighted.
    z = today's cumulative residual vs its fit-window distribution.
    z > 0: tenor high vs curve (cheap -> receive); z < 0: rich (pay).
    """
    oos = max(int(oos), 1)
    lv = curve.iloc[-(win + oos + 1):] * BP               # bp
    dx = lv.diff().dropna()
    fit = dx.iloc[:-oos]
    w = ew_weights(len(fit), hl)
    mu, sd, corr = weighted_moments(fit, w)
    eigval, eigvec = sorted_eig(corr, curve.columns)
    L = eigvec[:, :n_pc]
    zx = ((dx - mu) / sd).values                          # same mu/sd through today
    resid_bp = pd.DataFrame((zx - zx @ L @ L.T) * sd.values, index=dx.index,
                            columns=curve.columns)
    cum = resid_bp.cumsum()
    cf = cum.iloc[:len(fit)]
    z = (cum.iloc[-1] - cf.mean()) / cf.std()
    names = ["Level", "Slope", "Curvature"] + [f"PC{i}" for i in range(4, len(eigval) + 1)]
    load_bp = pd.DataFrame(eigvec * sd.values[:, None], index=curve.columns,
                           columns=names[:len(eigval)])     # bp move per 1sd PC
    return dict(
        explained=pd.Series(eigval / eigval.sum(), index=names[:len(eigval)]),
        loadings=load_bp.iloc[:, :n_pc],
        cum_resid=cum,
        fit_end=fit.index[-1],
        n_eff=float(w.sum() ** 2 / (w ** 2).sum()),
        z=z,
        half_life=cum.apply(half_life),
        levels=lv,
    )


def pca_fly(rv: dict, wings: tuple[int, int], belly: int) -> dict:
    """Belly +1, wings weighted so the fly has zero PC1 and PC2 exposure."""
    a, c = wings
    Lb = rv["loadings"].iloc[:, :2]
    A = np.array([[Lb.loc[a].iloc[0], Lb.loc[c].iloc[0]],
                  [Lb.loc[a].iloc[1], Lb.loc[c].iloc[1]]])
    wa, wc = np.linalg.solve(A, -Lb.loc[belly].values)
    lv = rv["levels"]
    fly = wa * lv[a] + lv[belly] + wc * lv[c]
    z = (fly.iloc[-1] - fly.mean()) / fly.std()
    return dict(weights={a: wa, belly: 1.0, c: wc}, series=fly, z=float(z),
                half_life=half_life(fly))


def rv_table(rv: dict, thr: float) -> pd.DataFrame:
    lv = rv["levels"].iloc[-1]
    df = pd.DataFrame({
        "Tenor": [f"{t}y" for t in rv["z"].index],
        "Rate (%)": (lv / BP).values,
        "Cum resid (bp)": rv["cum_resid"].iloc[-1].values,
        "z-score": rv["z"].values,
        "Half-life (d)": rv["half_life"].values,
    })
    df["Signal"] = np.where(df["z-score"] >= thr, "Cheap → receive",
                   np.where(df["z-score"] <= -thr, "Rich → pay", "–"))
    return df


# --------------------------------------------------------------------------
# WALK-FORWARD SIGNAL BACKTEST
# --------------------------------------------------------------------------
BT_H = 21          # forecast horizon (days) for IC / hit rate
BT_IC_WIN = 504    # trailing window for walk-forward IC weights
BT_REBAL = 21      # re-estimate IC weights every N days
BT_COST = {"outright": 0.25, "cross": 0.50, "steepener": 0.50}  # bp per unit of DV01 traded


def trade_pnl(f: pd.DataFrame, name: str) -> pd.Series:
    """Daily P&L in bp per 1 DV01 of the trade (long = as named), incl. carry/roll accrual."""
    kind, c = _trade_meta(name)
    if kind == "outright":
        return f[f"{c}_10y"].diff() * BP + f[f"{c}_cr3m"].shift(1) / 63
    if kind == "cross":
        b = f[f"{c}_beta"].shift(1)
        return (f[f"{c}_10y"].diff() - b * f.usd_10y.diff()) * BP \
            + (f[f"{c}_cr3m"] - f[f"{c}_beta"] * f.usd_cr3m).shift(1) / 63
    if kind == "steepener":
        return f[f"{c}_10s30s"].diff()
    raise ValueError(name)


def _position(x: pd.Series, cap: float = CLIP) -> pd.Series:
    return x.clip(-cap, cap).fillna(0.0)


def _strategy_stats(pos: pd.Series, pnl: pd.Series, cost_bp: float) -> dict:
    """Daily rebalanced: yesterday's position earns today's P&L, minus cost on turnover."""
    turn = pos.diff().abs().fillna(pos.abs())
    daily = (pos.shift(1) * pnl).fillna(0) - cost_bp * turn
    live = pos.shift(1).abs().gt(0) | daily.ne(0)
    d = daily[live.cumsum() > 0]
    yrs = max(len(d) / 252, 1e-9)
    sd = d.std()
    cd = d.cumsum()
    return dict(sharpe=float(d.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
                pnl_bp_yr=float(d.sum() / yrs),
                max_dd_bp=float((cd - cd.cummax()).min()) if len(cd) else np.nan,
                turnover_yr=float(turn[d.index].sum() / yrs),
                cum=daily.cumsum())


def _ic_stats(x: pd.Series, fwd: pd.Series, h: int) -> dict:
    v = x.notna() & fwd.notna() & np.isfinite(x)
    n = int(v.sum())
    if n < 3 * h:
        return dict(ic=np.nan, t=np.nan, hit=np.nan, n=n)
    ic = float(x[v].rank().corr(fwd[v].rank()))
    n_eff = n / h                                   # overlapping forward windows
    t = ic * np.sqrt(max(n_eff - 2, 1)) / np.sqrt(max(1 - ic ** 2, 1e-9))
    nz = v & x.ne(0) & fwd.ne(0)
    hit = float((np.sign(x[nz]) == np.sign(fwd[nz])).mean()) if nz.any() else np.nan
    return dict(ic=ic, t=t, hit=hit, n=n)


def _keep(ic: float, t: float, sharpe: float) -> str:
    if not np.isfinite(ic):
        return "n/a"
    if ic > 0 and t >= 2 and sharpe > 0:
        return "Keep"
    if ic > 0:
        return "Weak"
    return "Drop"


def trade_pnl_tenor(f: pd.DataFrame, c: str, t: int) -> pd.Series:
    """Daily P&L in bp per 1 DV01 of paying the t-year swap, incl. carry/roll accrual."""
    return f[f"{c}_t{t}"].diff() * BP + f[f"{c}_t{t}_cr3m"].shift(1) / 63


def _bt_one(w: dict, S: pd.DataFrame, pnl: pd.Series, kind: str, h: int, ic_win: int,
            rebal: int, cost_mult: float) -> dict:
    """Walk-forward evaluation of one strategy's signals and two composites.

    Per signal (oriented by the sign of its hand weight, so > 0 should mean the
    payer side makes money): rank IC vs the forward h-day P&L, t-stat with an
    overlap-adjusted sample (n/h), hit rate, and the Sharpe of a daily-rebalanced
    position = clip(signal, ±2) net of costs.
    Composites: 'Hand weights' = the live score; 'IC-weighted (WF)' = weights
    re-estimated every `rebal` days from the trailing `ic_win` days of realised
    IC, with an h-day embargo so no forward return leaks in.
    Conviction stats: on days with |score| >= CONVICTION, the forward h-day P&L
    of trading in the score's direction (pay if > 0, receive if < 0).
    """
    cost = BT_COST.get(kind, 0.5) * cost_mult
    idx = S.index
    fwd = pnl.rolling(h).sum().shift(-h)            # P&L over t+1 .. t+h
    O = S.mul(pd.Series({k: np.sign(v) for k, v in w.items()}))

    rows = []
    for k in O.columns:
        ics = _ic_stats(O[k], fwd, h)
        st_ = _strategy_stats(_position(O[k]), pnl, cost)
        rows.append(dict(Signal=k, Weight=w[k], IC=ics["ic"], t=ics["t"], Hit=ics["hit"],
                         Sharpe=st_["sharpe"], **{"P&L bp/yr": st_["pnl_bp_yr"],
                                                  "Turnover/yr": st_["turnover_yr"]},
                         Verdict=_keep(ics["ic"], ics["t"], st_["sharpe"])))
    sig_tbl = pd.DataFrame(rows)

    score = sum(S[k].mul(w[k]).clip(-CLIP, CLIP).fillna(0) for k in S.columns)
    pos_hand = (score / 2).clip(-CLIP, CLIP)

    scale = sum(abs(v) for v in w.values())
    Oc = O.clip(-CLIP, CLIP)
    pos_ic = pd.Series(0.0, index=idx)
    wts_hist = []
    first = ic_win + h
    for i in range(first, len(idx), rebal):
        lo, hi = max(0, i - h - ic_win), i - h      # embargo: fwd(t) needs t+h <= i
        ics = {k: _ic_stats(O[k].iloc[lo:hi], fwd.iloc[lo:hi], h)["ic"] for k in O.columns}
        raw = {k: max(v, 0.0) if np.isfinite(v) else 0.0 for k, v in ics.items()}
        tot = sum(raw.values())
        wk = {k: scale * v / tot for k, v in raw.items()} if tot > 0 else {k: 0.0 for k in raw}
        wts_hist.append((idx[i], wk))
        seg = slice(i, min(i + rebal, len(idx)))
        comp = sum(Oc[k].iloc[seg].fillna(0) * wk[k] for k in O.columns)
        pos_ic.iloc[seg] = (comp / 2).clip(-CLIP, CLIP).values

    mask = pd.Series(np.arange(len(idx)) >= first, index=idx)   # same OOS period for both
    comps = {lab: _strategy_stats(pos.where(mask, 0.0), pnl, cost)
             for lab, pos in [("Hand weights", pos_hand), ("IC-weighted (WF)", pos_ic)]}
    last_w = wts_hist[-1][1] if wts_hist else {}
    sig_tbl["WF weight (now)"] = sig_tbl.Signal.map(last_w)

    # conviction stats (hand score, no fitting -> full sample with a realised forward)
    now = float(score.iloc[-1])
    on = (score.abs() >= CONVICTION) & fwd.notna()
    dir_fwd = (np.sign(score) * fwd)[on]
    same = on & (np.sign(score) == np.sign(now)) if now != 0 else on & False
    conv = dict(days=int(on.sum()),
                hit=float((dir_fwd > 0).mean()) if on.any() else np.nan,
                avg=float(dir_fwd.mean()) if on.any() else np.nan,
                hit_same=float(((np.sign(now) * fwd)[same] > 0).mean()) if same.any() else np.nan,
                avg_same=float((np.sign(now) * fwd)[same].mean()) if same.any() else np.nan,
                days_same=int(same.sum()))
    return dict(signals=sig_tbl, composites=comps, kind=kind, score_now=now, conv=conv,
                score=score, oos_start=idx[first] if first < len(idx) else None)


def backtest(f: pd.DataFrame, h: int = BT_H, ic_win: int = BT_IC_WIN,
             rebal: int = BT_REBAL, cost_mult: float = 1.0) -> dict[str, dict]:
    """Walk-forward backtest of every scorecard strategy (10y outrights, crosses, curves)."""
    out = {}
    for name, (w, S) in signal_history(f).items():
        kind, _ = _trade_meta(name)
        r = _bt_one(w, S, trade_pnl(f, name), kind, h, ic_win, rebal, cost_mult)
        r["label"] = strategy_label(name, r["score_now"])
        out[neutral_name(name)] = r
    return out


def backtest_proposed(f: pd.DataFrame, props: list[dict], h: int = BT_H, ic_win: int = BT_IC_WIN,
                      rebal: int = BT_REBAL, cost_mult: float = 1.0) -> dict[str, dict]:
    """Walk-forward backtest of the proposed strategies, with their own tenor signals."""
    hist = signal_history(f)
    out = {}
    for sp in props:
        if sp["kind"] == "outright":
            c, t = sp["ccy"], sp["tenor"]
            sig = tenor_signals(f, c, t)
            w = {k: wt for k, (wt, _) in sig.items()}
            S = pd.DataFrame({k: v for k, (_, v) in sig.items()}, index=f.index)
            r = _bt_one(w, S, trade_pnl_tenor(f, c, t), "outright", h, ic_win, rebal, cost_mult)
        else:
            w, S = hist[sp["trade"]]
            r = _bt_one(w, S, trade_pnl(f, sp["trade"]), sp["kind"], h, ic_win, rebal, cost_mult)
        r["label"] = sp["label"]
        out[sp["label"]] = r
    return out


def money_pnl(bt: dict[str, dict], dv01: dict[str, float],
              composite: str = "Hand weights") -> tuple[pd.DataFrame, pd.Series]:
    """Convert bp-per-DV01 backtest P&L into money with the trader's DV01 per strategy.

    One position unit = 1 x DV01 (composites go up to 2 units at full conviction).
    Returns a per-strategy table and the out-of-sample daily P&L of the whole book
    (sum of strategies, same DV01s) for the chosen composite.
    """
    rows, book = [], []
    for name, r in bt.items():
        k = dv01.get(name, 0.0) or 0.0
        cv = r["conv"]
        row = {"Trade": name, "Strategy": r.get("label", name), "DV01 per bp": k}
        for lab, short in [("Hand weights", "Hand"), ("IC-weighted (WF)", "IC-WF")]:
            c = r["composites"][lab]
            row[f"{short} P&L/yr"] = c["pnl_bp_yr"] * k
            row[f"{short} max DD"] = c["max_dd_bp"] * k
        row["Avg fwd |s|≥1.5"] = cv["avg"] * k if np.isfinite(cv["avg"]) else np.nan
        row["Avg fwd same side"] = cv["avg_same"] * k if np.isfinite(cv["avg_same"]) else np.nan
        rows.append(row)
        cum = r["composites"][composite]["cum"]
        if r["oos_start"] is not None:
            cum = cum.loc[r["oos_start"]:]
        book.append(cum.diff().fillna(0) * k)
    daily = pd.concat(book, axis=1).fillna(0).sum(axis=1) if book else pd.Series(dtype=float)
    return pd.DataFrame(rows), daily


def book_stats(daily: pd.Series) -> dict:
    if daily.empty:
        return dict(pnl_yr=np.nan, sharpe=np.nan, max_dd=np.nan, total=np.nan)
    cum = daily.cumsum()
    sd = daily.std()
    return dict(pnl_yr=float(daily.sum() / max(len(daily) / 252, 1e-9)),
                sharpe=float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
                max_dd=float((cum - cum.cummax()).min()), total=float(cum.iloc[-1]))


def backtest_summary(bt: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, r in bt.items():
        t = r["signals"].dropna(subset=["IC"])
        best = t.loc[t.IC.idxmax()] if len(t) else None
        worst = t.loc[t.IC.idxmin()] if len(t) else None
        cv = r["conv"]
        rows.append({
            "Trade": name,
            "Strategy": r.get("label", name),
            "Score now": r["score_now"],
            "Sharpe hand": r["composites"]["Hand weights"]["sharpe"],
            "Sharpe IC-WF": r["composites"]["IC-weighted (WF)"]["sharpe"],
            "Hit |s|≥1.5": cv["hit"],
            "Avg fwd bp |s|≥1.5": cv["avg"],
            "Hit same side": cv["hit_same"],
            "Days same side": cv["days_same"],
            "Best signal": f"{best.Signal} (IC {best.IC:+.2f})" if best is not None else "–",
            "Worst signal": f"{worst.Signal} (IC {worst.IC:+.2f})" if worst is not None else "–",
            "Keep / Weak / Drop": "/".join(str((t.Verdict == v).sum()) for v in ["Keep", "Weak", "Drop"]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# MARKET COMMENT (rule-based, generated from the signals above)
# --------------------------------------------------------------------------
SIG_FR = {
    "10y momentum (vol-adj)": "momentum du 10y",
    "2y policy repricing": "repricing de la banque centrale (2y)",
    "10s30s steepening (term premium)": "dynamique du 10s30s (term premium)",
    "Breakeven momentum": "dynamique des breakevens",
    "Carry+roll / vol (3m)": "carry + roll",
    "Valuation (1y z, mean-rev)": "valorisation (niveau vs 1 an)",
    "France stress (flight to Bunds)": "stress France (fuite vers le Bund)",
    "Beta-adj residual momentum": "momentum du résidu beta-ajusté",
    "2y differential momentum": "dynamique du différentiel 2y",
    "Relative carry+roll / vol": "carry relatif",
    "Spread valuation (1y z)": "valorisation du spread",
    "France stress": "stress France",
    "Rate momentum (vol-adj)": "momentum du taux",
    "10s30s momentum": "momentum du 10s30s",
    "10s30s valuation (1y z)": "valorisation du 10s30s",
    "Swap spread momentum (govie supply)": "swap spreads / offre de govies",
}
STD_FLIES = [(2, 5, 10), (5, 10, 30), (10, 20, 30), (2, 10, 30)]
CONVICTION = 1.5


def _trade_meta(name: str) -> tuple[str, str]:
    """('outright'|'cross'|'steepener', ccy key) from a scorecard trade name."""
    import re
    if mt := re.match(r"Pay (\w+) / Rec USD", name):
        return "cross", mt.group(1).lower()
    if mt := re.match(r"(\w+) 10s30s steepener", name):
        return "steepener", mt.group(1).lower()
    for c, cfg in CCYS.items():
        if name == f"Pay {cfg['name']} 10y":
            return "outright", c
    return "other", ""


def _why(df: pd.DataFrame, side: int = 1) -> str:
    """Main drivers of a score; side = -1 reads the score from the receiver's side."""
    d = df.assign(Contribution=df.Contribution * side)
    pos = d[d.Contribution > 0.05].nlargest(2, "Contribution")
    neg = d[d.Contribution < -0.05].nsmallest(1, "Contribution")
    txt = " et ".join(SIG_FR.get(x, x) for x in pos.Signal) or "aucun signal fort"
    if len(neg):
        txt += f" ; principal frein : {SIG_FR.get(neg.Signal.iloc[0], neg.Signal.iloc[0])}"
    return txt


def _execution(name: str, f: pd.DataFrame) -> str:
    kind, c = _trade_meta(name)
    if kind == "outright":
        vol = f[f"{c}_vol"].iloc[-1]
        cr = f[f"{c}_cr3m"].iloc[-1]
        return (f"payer le 10y {CCYS[c]['name']} à {f[f'{c}_10y'].iloc[-1]:.3f}% ; "
                f"carry+roll 3m {cr:+.1f}bp ; vol réalisée {vol:.0f}bp/an, "
                f"stop indicatif ~1σ mensuel ≈ {vol / np.sqrt(12):.0f}bp")
    if kind == "cross":
        b = f[f"{c}_beta"].iloc[-1]
        rv = f[f"{c}_resid"].diff().rolling(VOL_WIN).std().iloc[-1] * np.sqrt(21)
        return (f"payer 1 DV01 de 10y {c.upper()} contre recevoir {b:.2f} DV01 de 10y USD "
                f"(beta 60j) ; spread actuel {f[f'{c}_usd_10y'].iloc[-1]:+.0f}bp ; "
                f"stop indicatif ~1σ mensuel du résidu ≈ {rv:.0f}bp")
    if kind == "steepener":
        return (f"recevoir 10y / payer 30y {c.upper()} en DV01-neutre ; "
                f"10s30s actuel {f[f'{c}_10s30s'].iloc[-1]:+.0f}bp")
    return ""


def _exec_tenor(r, side: int) -> str:
    verb = "payer" if side > 0 else "recevoir"
    cr = r.CR3m * side
    return (f"{verb} le {r.Tenor}y {r.Currency} à {r.Rate:.3f}% ; carry+roll 3m {cr:+.1f}bp "
            f"pour ce sens ; vol réalisée {r.Vol:.0f}bp/an, stop indicatif ~1σ mensuel ≈ "
            f"{r.Vol / np.sqrt(12):.0f}bp")


def _exec_cross(name: str, f: pd.DataFrame, side: int) -> str:
    kind, c = _trade_meta(name)
    if kind == "cross":
        b = f[f"{c}_beta"].iloc[-1]
        rv = f[f"{c}_resid"].diff().rolling(VOL_WIN).std().iloc[-1] * np.sqrt(21)
        a, z = ("payer", "recevoir") if side > 0 else ("recevoir", "payer")
        return (f"{a} 1 DV01 de 10y {c.upper()} contre {z} {b:.2f} DV01 de 10y USD (beta 60j) ; "
                f"spread {f[f'{c}_usd_10y'].iloc[-1]:+.0f}bp ; stop ~1σ mensuel du résidu ≈ {rv:.0f}bp")
    if kind == "steepener":
        lab = ("steepener : recevoir 10y / payer 30y" if side > 0
               else "flattener : payer 10y / recevoir 30y")
        return f"{lab} {c.upper()} en DV01-neutre ; 10s30s actuel {f[f'{c}_10s30s'].iloc[-1]:+.0f}bp"
    return ""


def _cross_label(name: str, side: int) -> str:
    kind, c = _trade_meta(name)
    if kind == "cross":
        return f"Pay {c.upper()} / Rec USD 10y" if side > 0 else f"Rec {c.upper()} / Pay USD 10y"
    if kind == "steepener":
        return f"{c.upper()} 10s30s {'steepener' if side > 0 else 'flattener'}"
    return name


def market_comment(f: pd.DataFrame, px: pd.DataFrame, thr: float = 2.0,
                   pca_win: int = PCA_WIN, n_pc: int = N_PC, oos: int = PCA_OOS,
                   hl: float = PCA_HL, max_hl: float = 30.0) -> dict:
    """Systematic market comment in French: preferred trade, alternatives,
    trades to avoid, curve RV ideas, risks. Generated from the signals only."""
    sc = scorecard(f)
    dm, dd = directional_matrix(f)
    sections: list[tuple[str, list[str]]] = []

    # --- outright ideas, every currency x tenor, both sides
    def tenor_ideas(side: int) -> list[str]:
        sub = dm[dm.Score * side >= CONVICTION]
        if sub.empty:
            return []
        out = []
        for c, g in sub.assign(a=sub.Score * side).sort_values("a", ascending=False).groupby("ccy", sort=False):
            r = g.iloc[0]
            n_all = int((dm[dm.ccy == c].Score * side >= CONVICTION).sum())
            extra = (f" Signal sur {n_all}/{(dm.ccy == c).sum()} ténors : c'est un mouvement de niveau, "
                     "le ténor le mieux noté est le point d'entrée." if n_all >= 4 else
                     (" Autres ténors au-dessus du seuil : " + ", ".join(f"{x}y" for x in g.Tenor.iloc[1:]) + "."
                      if len(g) > 1 else ""))
            out.append((r.Score * side,
                        f"**{'Payer' if side > 0 else 'Recevoir'} {r.Tenor}y {r.Currency}** "
                        f"(score {r.Score:+.2f}) — {_why(dd[(c, r.Tenor)], side)}.{extra}"))
        return [t for _, t in sorted(out, key=lambda x: -x[0])][:4]

    pay_l, rec_l = tenor_ideas(1), tenor_ideas(-1)

    # --- cross-market and curve trades, both sides
    xr = []
    for name, df in sc.items():
        kind, _ = _trade_meta(name)
        if kind not in ("cross", "steepener"):
            continue
        tot = float(df.Contribution.sum())
        if abs(tot) >= CONVICTION:
            sd = 1 if tot > 0 else -1
            xr.append((abs(tot), name, sd,
                       f"**{_cross_label(name, sd)}** (score {tot:+.2f}) — {_why(df, sd)}. "
                       f"Exécution : {_exec_cross(name, f, sd)}."))
    xr.sort(key=lambda x: -x[0])

    # --- headline: strongest idea across everything
    cands = [(abs(r.Score), "tenor", r) for r in dm.itertuples() if abs(r.Score) >= CONVICTION]
    cands += [(a, "x", (n, sd)) for a, n, sd, _ in xr]
    if cands:
        a, kind, obj = max(cands, key=lambda x: x[0])
        if kind == "tenor":
            sd = 1 if obj.Score > 0 else -1
            headline = (f"Trade privilégié : **{'payer' if sd > 0 else 'recevoir'} le {obj.Tenor}y "
                        f"{obj.Currency}** (score {obj.Score:+.2f}). Moteurs : {_why(dd[(obj.ccy, obj.Tenor)], sd)}. "
                        f"Exécution : {_exec_tenor(obj, sd)}.")
        else:
            n, sd = obj
            headline = (f"Trade privilégié : **{_cross_label(n, sd)}** (score {a * sd:+.2f}). "
                        f"Moteurs : {_why(sc[n], sd)}. Exécution : {_exec_cross(n, f, sd)}.")
        if a < 2.5:
            headline += " Conviction modérée : taille réduite tant que |score| reste sous 2,5."
    else:
        headline = ("Pas de signal directionnel fort sur aucun ténor ni cross. Rester léger sur "
                    "la duration ; privilégier la RV de courbe.")

    sections.append(("Payer", pay_l or ["Aucun ténor au-dessus du seuil de conviction (+1,5)."]))
    sections.append(("Recevoir", rec_l or ["Aucun ténor sous le seuil de conviction (−1,5)."]))
    sections.append(("Cross-marché et pente", [t for *_, t in xr][:4] or
                     ["Aucun cross ni trade de pente au-delà de ±1,5."]))
    best_name = max(xr, key=lambda x: x[0])[1] if xr else ""

    # --- curve RV
    rv_lines = []
    for c in CCYS:
        cv = curve_frame(px, c)
        if cv.shape[1] < 4 or len(cv) < pca_win // 2:
            continue
        rv = pca_rv(cv, pca_win, min(n_pc, cv.shape[1] - 1), oos, hl)
        tbl = rv_table(rv, thr)
        hits = tbl[(tbl["z-score"].abs() >= thr) & (tbl["Half-life (d)"] <= max_hl)]
        for r in hits.itertuples(index=False):
            side = "recevoir" if r[3] > 0 else "payer"
            state = "cheap" if r[3] > 0 else "rich"
            rv_lines.append((abs(r[3]),
                f"**{side} le {r.Tenor} {CCYS[c]['name']}** ({state}, z {r[3]:+.2f}, "
                f"demi-vie {r[4]:.0f}j) — "
                + ("couvrir level/slope avec les ailes voisines."
                   if min(cv.columns) < int(r.Tenor[:-1]) < max(cv.columns)
                   else "ténor extrême : couvrir contre les deux ténors adjacents.")))
        if rv["loadings"].shape[1] >= 2:
            best_fly = None
            for a, b, d in STD_FLIES:
                if {a, b, d} <= set(cv.columns):
                    fl = pca_fly(rv, (a, d), b)
                    if np.isfinite(fl["half_life"]) and fl["half_life"] <= max_hl * 1.5 \
                            and abs(fl["z"]) >= thr and (best_fly is None or abs(fl["z"]) > abs(best_fly[1]["z"])):
                        best_fly = ((a, b, d), fl)
            if best_fly:
                (a, b, d), fl = best_fly
                w = fl["weights"]
                side = "recevoir" if fl["z"] > 0 else "payer"
                rv_lines.append((abs(fl["z"]),
                    f"**Fly {a}s{b}s{d}s {CCYS[c]['name']}** : {side} le ventre {b}y "
                    f"({w[a]:+.2f}×{a}y / +1×{b}y / {w[d]:+.2f}×{d}y en DV01, PC1/PC2-neutre) — "
                    f"z {fl['z']:+.2f}, demi-vie {fl['half_life']:.0f}j."))
    rv_lines = [t for _, t in sorted(rv_lines, key=lambda x: -x[0])][:5]
    sections.append(("Relative value de courbe (PCA)", rv_lines or
                     [f"Aucun ténor ni fly au-delà de |z| {thr:g} avec une demi-vie ≤ {max_hl:.0f}j."]))

    # --- risks
    risks = []
    oz = _last(zscore(f.oat_bund))
    if np.isfinite(oz) and oz > 1.0:
        risks.append(f"Stress France (OAT/Bund {f.oat_bund.iloc[-1]:.0f}bp, z {oz:.1f}) : un "
                     "flight-to-quality ferait baisser Bunds et swaps EUR — risque pour tout payeur EUR, "
                     "soutien pour les receveurs.")
    mv = f.move.diff(MOM_WIN).iloc[-1]
    if np.isfinite(mv) and mv > 5:
        risks.append(f"MOVE en hausse ({mv:+.0f} pts sur 20j) : réduire la taille ou passer par des options.")
    kind, c = _trade_meta(best_name)
    if kind == "cross":
        b, blt = f[f"{c}_beta"].iloc[-1], f[f"{c}_beta_lt"].iloc[-1]
        if np.isfinite(b) and np.isfinite(blt) and abs(b - blt) > ALERTS["beta_shift"]:
            risks.append(f"Beta {c.upper()}/USD instable ({b:.2f} vs {blt:.2f} en moyenne 1 an) : "
                         "le ratio de couverture peut dériver, recalibrer chaque semaine.")
    risks.append("Vérifier le calendrier (banques centrales, adjudications, données d'inflation) "
                 "avant d'entrer : les signaux ne voient pas les événements.")
    sections.append(("Risques", risks))

    return dict(headline=headline, sections=sections,
                disclaimer="Commentaire généré automatiquement à partir des signaux du monitor ; "
                           "ce n'est pas une recommandation d'investissement.")


def _score_series(sig: dict[str, tuple[float, pd.Series]]) -> pd.Series:
    return sum(v.mul(w).clip(-CLIP, CLIP).fillna(0) for w, v in sig.values())


def proposed_strategies(f: pd.DataFrame, max_n: int = 6, target_mult: float = 2.0) -> list[dict]:
    """Strategies the comment proposes (|score| >= CONVICTION), with what to chart.

    'series' is the traded quantity, oriented so the payer side (pay / pay X-rec USD /
    steepener) gains when it rises. Stop = 1 monthly sigma against the position,
    target = target_mult x that distance in favour. Units: % for outright rates,
    bp for spreads and curves.
    """
    dm, _ = directional_matrix(f)
    out = []
    # best tenor per currency and side
    for side in (1, -1):
        sub = dm[dm.Score * side >= CONVICTION]
        for c, g in sub.assign(a=sub.Score * side).sort_values("a", ascending=False).groupby("ccy", sort=False):
            r = g.iloc[0]
            t = int(r.Tenor)
            sig = tenor_signals(f, c, t)
            sm = r.Vol / np.sqrt(12) / BP          # monthly sigma in %
            out.append(dict(
                label=f"{'Pay' if side > 0 else 'Receive'} {t}y {CCYS[c]['name']}",
                ccy=c, tenor=t, trade=None, kind="outright", side=side, score=float(r.Score), conv=abs(float(r.Score)),
                series=f[f"{c}_t{t}"], unit="%", series_name=f"{t}y {CCYS[c]['name']} (%)",
                score_hist=_score_series(sig), entry=float(r.Rate),
                stop=float(r.Rate - side * sm), target=float(r.Rate + side * target_mult * sm),
                sigma=sm * BP))
    for name, (w, S) in signal_history(f).items():
        kind, c = _trade_meta(name)
        if kind not in ("cross", "steepener"):
            continue
        sh = sum(S[k].mul(w[k]).clip(-CLIP, CLIP).fillna(0) for k in S.columns)
        sc_ = float(sh.iloc[-1])
        if abs(sc_) < CONVICTION:
            continue
        side = 1 if sc_ > 0 else -1
        if kind == "cross":
            ser_ = f[f"{c}_resid"]
            sname = f"{c.upper()} 10y − β·USD 10y, cumulated (bp)"
        else:
            ser_ = f[f"{c}_10s30s"]
            sname = f"{c.upper()} 10s30s (bp)"
        sm = float(ser_.diff().rolling(VOL_WIN).std().iloc[-1] * np.sqrt(21))
        e = float(ser_.iloc[-1])
        out.append(dict(label=strategy_label(name, sc_).replace(" (beta-adj)", ""), kind=kind,
                        ccy=c, tenor=None, trade=name,
                        side=side, score=sc_, conv=abs(sc_), series=ser_, unit="bp",
                        series_name=sname, score_hist=sh, entry=e, stop=e - side * sm,
                        target=e + side * target_mult * sm, sigma=sm))
    out.sort(key=lambda x: -x["conv"])
    return out[:max_n]


def comment_markdown(mc: dict) -> str:
    out = [mc["headline"], ""]
    for title, items in mc["sections"]:
        out.append(f"**{title}**")
        out += [f"- {x}" for x in items]
        out.append("")
    out.append(f"_{mc['disclaimer']}_")
    return "\n".join(out)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def main() -> None:
    import plotly.express as px_
    import streamlit as st

    st.set_page_config(page_title="G4 Payer Monitor", layout="wide")
    st.title("G4 10y Payer Monitor — SOFR · €STR · SONIA · CORRA")

    with st.sidebar:
        bbg_ok = bloomberg_available()
        sources = (["Bloomberg (xbbg)"] if bbg_ok else []) + ["Public bond data", "Demo"]
        source = st.radio("Data source", sources, index=0,
                          help="Public bond data: government yields and fixings from FRED, ECB, "
                               "Bank of England and Bank of Canada, as proxies for OIS swaps.")
        years = st.slider("History (years)", 1, 10, 5, help="The backtest needs at least 3 years.")
        end = st.date_input("End date", dt.date.today())
        show_n = st.slider("Chart window (days)", 60, 750, 250)
        st.markdown("**Curve RV (PCA)**")
        pca_win = st.slider("PCA window (days)", 126, 504, PCA_WIN, step=21)
        n_pc = st.slider("PCs kept", 1, 4, N_PC)
        pca_oos = st.slider("Out-of-sample days (fit ends N days ago)", 5, 63, PCA_OOS)
        pca_hl = st.select_slider("EW half-life for loadings (days, 0 = equal)",
                                  [0, 21, 42, 63, 84, 126, 189, 252], value=PCA_HL)
        z_thr = st.slider("Signal |z| threshold", 1.0, 3.0, 2.0, step=0.25)
        st.markdown("**Sizing**")
        dv01_default = st.number_input("DV01 per strategy (amount per bp)", min_value=0.0,
                                       value=10_000.0, step=1_000.0, format="%.0f", key="dv01")
        ccy_lab = st.selectbox("P&L currency", ["EUR", "USD", "GBP", "CAD"], key="dv01_ccy")

    start = end - dt.timedelta(days=365 * years + 30)

    @st.cache_data(ttl=6 * 3600, show_spinner="Loading data…")
    def load(src: str, s: dt.date, e: dt.date):
        if src == "Demo":
            return make_demo(s, e)
        if src == "Public bond data":
            return fetch_public(s, e)
        return fetch_bloomberg(s, e)

    try:
        px, missing = load(source, start, end)
    except Exception as e:  # noqa: BLE001
        st.error(f"Data load failed: {e}")
        st.stop()
    if missing:
        if source == "Public bond data":
            st.info(" · ".join(missing))
        else:
            st.info(f"Tickers not returned (signals using them are skipped): {missing}")

    f = build_features(px)
    st.caption(f"As of {f.index[-1].date()} · source: {source}")
    if source == "Demo":
        st.warning("Synthetic demo data: levels, signals and backtest results are illustrative only, "
                   "not market data and not investment advice.")
    elif source == "Public bond data":
        st.warning("Public data: government bond yields (US Treasuries, euro area AAA curve, gilts, "
                   "GoC) stand in for OIS swaps, with real overnight fixings. Labels such as 'SOFR' "
                   "refer to that currency's curve. Not investment advice.")

    tab_mkt, tab_strat, tab_bt, tab_rv, tab_data = st.tabs(
        ["Market view", "Strategies", "Backtest", "Curve RV (PCA)", "Data"])

    with tab_mkt:
        st.subheader("Alerts")
        al = alerts(f)
        if al:
            for x in al:
                st.warning(x)
        else:
            st.success("No thresholds breached.")

        st.subheader("Market comment")
        try:
            mc = market_comment(f, px, z_thr, pca_win, n_pc, pca_oos, pca_hl)
            st.markdown(comment_markdown(mc))
        except Exception as e:  # noqa: BLE001
            st.info(f"Market comment unavailable: {e}")

        st.subheader("Proposed strategies — charts")
        props = proposed_strategies(f)
        if not props:
            st.info("No strategy above the conviction threshold.")
        else:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            cols_ = st.columns(2)
            for i, sp in enumerate(props):
                ww = f.index[-show_n:]
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=ww, y=sp["score_hist"].loc[ww], name="Score",
                                     marker_color=["#c0392b" if v > 0 else "#1e8449"
                                                   for v in sp["score_hist"].loc[ww]],
                                     opacity=0.25), secondary_y=True)
                fig.add_trace(go.Scatter(x=ww, y=sp["series"].loc[ww], name=sp["series_name"],
                                         line=dict(width=1.8)), secondary_y=False)
                fmt_ = (lambda v: f"{v:.3f}%") if sp["unit"] == "%" else (lambda v: f"{v:+.1f}bp")
                for lvl, lab, dash in [(sp["entry"], "entry", "solid"), (sp["stop"], "stop", "dot"),
                                       (sp["target"], "target", "dash")]:
                    fig.add_hline(y=lvl, line_dash=dash, line_width=1,
                                  annotation_text=f"{lab} {fmt_(lvl)}", annotation_position="left")
                risk_m = sp["sigma"] * dv01_default
                fig.update_layout(height=300, title=f"{sp['label']} · score {sp['score']:+.2f} · "
                                                    f"risk {risk_m:,.0f} / reward {2 * risk_m:,.0f} {ccy_lab}",
                                  showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                fig.update_yaxes(title_text=sp["series_name"], secondary_y=False)
                fig.update_yaxes(title_text="Score", secondary_y=True, showgrid=False)
                cols_[i % 2].plotly_chart(fig, use_container_width=True, key=f"prop_{i}")


    with tab_strat:
        st.subheader("Strategies, pay or receive, ranked by conviction")
        st.caption("Each strategy is shown in its preferred direction. Score > 0: payer / Pay X–Rec USD / "
                   "steepener; score < 0: receiver / Rec X–Pay USD / flattener. Contributions are read "
                   "from the side shown.")
        sc = scorecard(f)
        summ = summary_table(sc)
        st.dataframe(summ[["Strategy", "Score", "Conviction", "Read"]].style.format(
            {"Score": "{:+.2f}", "Conviction": "{:.2f}"}), hide_index=True, use_container_width=True)
        for tab, r in zip(st.tabs(list(summ.Strategy)), summ.itertuples()):
            with tab:
                st.dataframe(oriented_detail(sc[r.Trade], r.Score).style.format(
                    {"Weight": "{:+.1f}", "Raw": "{:+.2f}", "Contribution": "{:+.2f}"}),
                    hide_index=True, use_container_width=True)

        st.subheader("Directional by tenor (pay > 0, receive < 0)")
        dm, dd = directional_matrix(f)
        hm = dm.pivot(index="Currency", columns="Tenor", values="Score").reindex(
            [CCYS[c]["name"] for c in CCYS])
        hm.columns = [f"{t}y" for t in hm.columns]
        lim = max(3.0, float(np.nanmax(np.abs(hm.values))))
        fig_hm = px_.imshow(hm, text_auto="+.1f", color_continuous_scale="RdYlGn_r",
                            zmin=-lim, zmax=lim, aspect="auto")
        fig_hm.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                             coloraxis_colorbar=dict(title="Score"))
        st.plotly_chart(fig_hm, use_container_width=True, key="dir_hm")
        d1, d2 = st.columns(2)
        dc = d1.selectbox("Currency", list(CCYS), format_func=lambda c: CCYS[c]["name"], key="dir_c")
        dt_ = d2.selectbox("Tenor", [t for t in TENORS if (dc, t) in dd], index=2,
                           format_func=lambda t: f"{t}y", key="dir_t")
        row_ = dm[(dm.ccy == dc) & (dm.Tenor == dt_)].iloc[0]
        st.caption(f"{row_.Currency} {dt_}y · {row_.Rate:.3f}% · score {row_.Score:+.2f} → "
                   f"**{row_.Side}** · payer carry+roll 3m {row_.CR3m:+.1f}bp · vol {row_.Vol:.0f}bp/yr")
        st.dataframe(dd[(dc, dt_)].style.format({"Weight": "{:+.2f}", "Raw": "{:+.2f}",
                                                  "Contribution": "{:+.2f}"}),
                     hide_index=True, use_container_width=True)


    with tab_bt:
        # --- Walk-forward backtest
        st.subheader("Signal backtest (walk-forward) — proposed strategies")
        st.caption("Runs on the strategies proposed above, each with its own signals (tenor-specific "
                   "for outrights). Signals are oriented so positive IC means they predicted the payer "
                   "side's P&L. P&L in bp per 1 DV01, incl. carry/roll, net of costs. 'Hit |s|≥1.5': "
                   "share of past days with conviction where trading in the score's direction made "
                   "money over the horizon; 'same side' restricts to today's direction.")
        b1, b2, b3, b4, b5 = st.columns(5)
        bt_h = b1.select_slider("Horizon (days)", [5, 10, 21, 42, 63], value=BT_H, key="bt_h")
        bt_win = b2.select_slider("IC window (days)", [252, 378, 504, 756], value=BT_IC_WIN, key="bt_win")
        bt_reb = b3.select_slider("Re-weight every (days)", [5, 10, 21, 63], value=BT_REBAL, key="bt_reb")
        bt_cost = b4.select_slider("Cost multiplier", [0.0, 0.5, 1.0, 2.0, 3.0], value=1.0, key="bt_cost")
        bt_all = b5.toggle("Include all scorecard trades", value=False, key="bt_all")
        if len(f) < bt_win + bt_h + 252:
            st.info("History too short for a meaningful walk-forward test: raise 'History (years)'.")
        else:
            bt = backtest_proposed(f, props, bt_h, bt_win, bt_reb, bt_cost) if props else {}
            if bt_all:
                bt.update(backtest(f, bt_h, bt_win, bt_reb, bt_cost))
            if not bt:
                st.info("No proposed strategy to test.")
            else:
                st.markdown(f"**P&L in {ccy_lab}** — one position unit = 1 × DV01; "
                            "composites reach 2 units at full conviction. Edit DV01 per strategy below.")
                dv_tbl = pd.DataFrame({"Strategy": [bt[k].get("label", k) for k in bt],
                                       "DV01 per bp": [dv01_default] * len(bt)}, index=list(bt))
                dv_edit = st.data_editor(dv_tbl, hide_index=True, disabled=["Strategy"],
                                         use_container_width=True, key="dv01_editor",
                                         column_config={"DV01 per bp": st.column_config.NumberColumn(
                                             format="%.0f", min_value=0.0, step=1000.0)})
                dv_map = dict(zip(bt.keys(), dv_edit["DV01 per bp"].astype(float)))
                comp_sel = st.radio("Composite", ["Hand weights", "IC-weighted (WF)"], horizontal=True,
                                    key="bt_comp")
                mtab, book = money_pnl(bt, dv_map, comp_sel)
                bs = book_stats(book)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric(f"Book P&L / yr ({ccy_lab})", f"{bs['pnl_yr']:,.0f}")
                k2.metric("Book Sharpe", f"{bs['sharpe']:+.2f}")
                k3.metric(f"Book max drawdown ({ccy_lab})", f"{bs['max_dd']:,.0f}")
                k4.metric(f"Book total OOS ({ccy_lab})", f"{bs['total']:,.0f}")
                money_cols = [c for c in mtab.columns if c not in ("Trade", "Strategy")]
                st.dataframe(mtab.drop(columns="Trade").style.format(
                    {c: "{:,.0f}" for c in money_cols}, na_rep="–"),
                    hide_index=True, use_container_width=True)
                if not book.empty:
                    fb = px_.area(book.cumsum().rename(f"Book cumulative P&L ({ccy_lab})"),
                                  title=f"Book: cumulative out-of-sample P&L, {comp_sel} ({ccy_lab})")
                    fb.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fb, use_container_width=True, key="bt_book")

                st.markdown("**Signal diagnostics (bp per DV01)**")
                summ_bt = backtest_summary(bt)
                st.dataframe(summ_bt.drop(columns="Trade").style.format({
                    "Score now": "{:+.2f}", "Sharpe hand": "{:+.2f}", "Sharpe IC-WF": "{:+.2f}",
                    "Hit |s|≥1.5": "{:.0%}", "Avg fwd bp |s|≥1.5": "{:+.1f}", "Hit same side": "{:.0%}"},
                    na_rep="–"), hide_index=True, use_container_width=True)
                tr = st.selectbox("Strategy detail", list(bt.keys()), key="bt_trade")
                r = bt[tr]
                st.dataframe(r["signals"].style.format({
                    "Weight": "{:+.2f}", "IC": "{:+.3f}", "t": "{:+.2f}", "Hit": "{:.0%}",
                    "Sharpe": "{:+.2f}", "P&L bp/yr": "{:+.1f}", "Turnover/yr": "{:.0f}",
                    "WF weight (now)": "{:.2f}"}, na_rep="–"),
                    hide_index=True, use_container_width=True)
                cum = pd.DataFrame({k: v["cum"] for k, v in r["composites"].items()})
                if r["oos_start"] is not None:
                    cum = cum.loc[r["oos_start"]:]
                    cum = cum - cum.iloc[0]
                in_money = st.toggle(f"Show in {ccy_lab}", value=True, key="bt_money")
                unit = f"{ccy_lab} at DV01 {dv_map.get(tr, 0):,.0f}" if in_money else "bp per DV01"
                if in_money:
                    cum = cum * dv_map.get(tr, 0.0)
                fig = px_.line(cum, title=f"{r['label']}: cumulative P&L, {unit} (out-of-sample, "
                                          "long/short on the score)")
                fig.update_layout(height=320, legend_title=None, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, key="bt_cum")


    with tab_rv:
        # --- Curve RV per currency
        st.subheader("Curve RV — PCA residual z-scores")
        st.caption("PCA on normalised daily changes ((Δ − mean) / σ), fitted on a window ending "
                   "N days ago and applied through today. Residual = move not explained by the "
                   "kept PCs, cumulated. z > 0: tenor cheap vs curve (receive); z < 0: rich (pay).")
        for tab, c in zip(st.tabs([CCYS[c]["name"] for c in CCYS]), CCYS):
            with tab:
                cv = curve_frame(px, c)
                if cv.shape[1] < 4 or len(cv) < pca_win // 2:
                    st.warning("Not enough curve points / history for PCA.")
                    continue
                rv = pca_rv(cv, pca_win, min(n_pc, cv.shape[1] - 1), pca_oos, pca_hl)
                ev = rv["explained"]
                m = st.columns(4)
                for col, (nm, v) in zip(m, ev.iloc[:3].items()):
                    col.metric(f"{nm} var.", f"{v:.1%}")
                m[3].metric(f"Top {len(rv['loadings'].columns)} PCs",
                            f"{ev.iloc[:rv['loadings'].shape[1]].sum():.1%}",
                            f"eff. sample {rv['n_eff']:.0f}d", delta_color="off")

                tbl = rv_table(rv, z_thr)
                l, r = st.columns([1, 1])
                fig = px_.bar(tbl, x="Tenor", y="z-score", title="Residual z-score by tenor",
                              color=tbl["z-score"].abs() >= z_thr,
                              color_discrete_map={True: "#d62728", False: "#7f7f7f"})
                fig.add_hline(y=z_thr, line_dash="dot")
                fig.add_hline(y=-z_thr, line_dash="dot")
                fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                l.plotly_chart(fig, use_container_width=True, key=f"z_{c}")
                r.dataframe(tbl.style.format({"Rate (%)": "{:.3f}", "Cum resid (bp)": "{:+.1f}",
                                              "z-score": "{:+.2f}", "Half-life (d)": "{:.0f}"}),
                            hide_index=True, use_container_width=True)

                l2, r2 = st.columns(2)
                ld = rv["loadings"].copy()
                ld.index = [f"{t}y" for t in ld.index]
                fl = px_.line(ld, markers=True, title="PC loadings (bp per 1σ move)")
                fl.update_layout(height=300, legend_title=None, margin=dict(l=10, r=10, t=40, b=10))
                l2.plotly_chart(fl, use_container_width=True, key=f"ld_{c}")
                ten = r2.selectbox("Tenor residual", list(cv.columns), index=2,
                                   format_func=lambda t: f"{t}y", key=f"ten_{c}")
                fr_ = px_.line(rv["cum_resid"][ten], title=f"{ten}y cumulative residual (bp)")
                fr_.add_vline(x=rv["fit_end"], line_dash="dot")
                fr_.update_layout(height=255, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                r2.plotly_chart(fr_, use_container_width=True, key=f"res_{c}")

                with st.expander("Loading stability (rolling re-estimation every 5d)"):
                    rl = rolling_loadings(cv, pca_win, pca_hl, n_pc=min(3, cv.shape[1] - 1))
                    pc_sel = st.radio("PC", list(rl.keys()), horizontal=True, key=f"pcsel_{c}")
                    fs = px_.line(rl[pc_sel], title=f"{pc_sel} loadings over time (unit eigenvector)")
                    fs.update_layout(height=300, legend_title=None,
                                     margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fs, use_container_width=True, key=f"stab_{c}")
                    drift = rl[pc_sel].diff().abs().mean(axis=1).mean()
                    st.caption(f"Avg loading change per re-estimate: {drift:.3f}. "
                               "Large or erratic moves → lengthen the half-life.")

                st.markdown("**PC1/PC2-neutral butterfly**")
                ts = list(cv.columns)
                f1, f2, f3 = st.columns(3)
                w1 = f1.selectbox("Short wing", ts, index=ts.index(5) if 5 in ts else 0,
                                  format_func=lambda t: f"{t}y", key=f"w1_{c}")
                bl = f2.selectbox("Belly", ts, index=ts.index(10) if 10 in ts else 1,
                                  format_func=lambda t: f"{t}y", key=f"b_{c}")
                w2 = f3.selectbox("Long wing", ts, index=ts.index(30) if 30 in ts else len(ts) - 1,
                                  format_func=lambda t: f"{t}y", key=f"w2_{c}")
                if not (w1 < bl < w2):
                    st.info("Pick short wing < belly < long wing.")
                    continue
                if rv["loadings"].shape[1] < 2:
                    st.info("Keep at least 2 PCs for a neutral fly.")
                    continue
                fly = pca_fly(rv, (w1, w2), bl)
                wtxt = " ".join(f"{w:+.2f}×{t}y" for t, w in fly["weights"].items())
                st.write(f"Weights (bp-risk): {wtxt} · z {fly['z']:+.2f} · "
                         f"half-life {fly['half_life']:.0f}d")
                ff = px_.line(fly["series"], title=f"{w1}s{bl}s{w2}s PCA fly (bp)")
                ff.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(ff, use_container_width=True, key=f"fly_{c}")


    with tab_data:
        with st.expander("Raw data"):
            st.dataframe(px.tail(30))


if __name__ == "__main__":
    main()
