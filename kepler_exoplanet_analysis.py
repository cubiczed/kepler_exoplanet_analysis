"""
Usage:
    python kepler_exoplanet_analysis.py --limit 2000 --outdir results

Dependencies:
    pip install requests pandas numpy matplotlib scipy
"""
import argparse
import io
import math
import sys
import textwrap
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
import matplotlib.pyplot as plt

# -------------------------- Constants -------------------------- #
G = 6.67430e-11            # m^3 kg^-1 s^-2 
M_SUN = 1.98847e30         # kg
AU = 1.495978707e11        # m
DAY = 86400.0              # s

EXPECTED_SLOPE_LOG = 1.5   # slope of log T vs log r
EXPECTED_CONST = 4.0 * math.pi**2 / G  # expected slope of T^2 M vs r^3 (in SI)

# ----------------------- Helper Data Types --------------------- #
@dataclass
class FitResult:
    slope: float
    intercept: float
    r_value: float
    p_value: float
    stderr: float

# --------------------------- TAP Query ------------------------- #
def build_tap_query(limit: int) -> str:
    cols = [
        "pl_name", "hostname",
        "pl_orbper","pl_orbpererr1","pl_orbpererr2",
        "pl_orbsmax","pl_orbsmaxerr1","pl_orbsmaxerr2",
        "st_mass","st_masserr1","st_masserr2",
        "sy_snum","sy_pnum"
    ]
    select_cols = ",".join(cols)
    base = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = f"SELECT TOP {limit} {select_cols} FROM ps WHERE pl_orbper IS NOT NULL AND pl_orbsmax IS NOT NULL AND st_mass IS NOT NULL ORDER BY pl_name ASC"
    return f"{base}?query={requests.utils.quote(query)}&format=csv"

def fetch_dataframe(limit: int) -> pd.DataFrame:
    url = build_tap_query(limit)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

# ---------------------- Cleaning & Units ------------------------ #
def _symmetrize_uncert(pos: Optional[float], neg: Optional[float]) -> Optional[float]:
    """Convert asymmetric (+/-) uncertainties to a single sigma via mean absolute value."""
    if pd.isna(pos) and pd.isna(neg):
        return None
    vals = []
    if not pd.isna(pos):
        vals.append(abs(float(pos)))
    if not pd.isna(neg):
        vals.append(abs(float(neg)))
    if not vals:
        return None
    return float(np.nanmean(vals))

def clean_and_convert(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows missing any of the key quantities
    df = df.dropna(subset=["pl_orbper", "pl_orbsmax", "st_mass"]).reset_index(drop=True)

    # Symmetric uncertainties (may be NaN if unavailable)
    df["sigma_T_days"] = [
        _symmetrize_uncert(p, n) for p, n in zip(df["pl_orbpererr1"], df["pl_orbpererr2"])
    ]
    df["sigma_a_AU"] = [
        _symmetrize_uncert(p, n) for p, n in zip(df["pl_orbsmaxerr1"], df["pl_orbsmaxerr2"])
    ]
    df["sigma_M_sun"] = [
        _symmetrize_uncert(p, n) for p, n in zip(df["st_masserr1"], df["st_masserr2"])
    ]

    # Unit conversions to SI
    df["T_s"] = df["pl_orbper"].astype(float) * DAY
    df["a_m"] = df["pl_orbsmax"].astype(float) * AU
    df["M_kg"] = df["st_mass"].astype(float) * M_SUN

    df["sigma_T_s"] = df["sigma_T_days"] * DAY
    df["sigma_a_m"] = df["sigma_a_AU"] * AU
    df["sigma_M_kg"] = df["sigma_M_sun"] * M_SUN

    # Derived quantities
    df["T2"] = df["T_s"] ** 2
    df["a3"] = df["a_m"] ** 3
    df["T2M"] = df["T2"] * df["M_kg"]

    # Uncertainty propagation (first-order)
    # sigma(T^2) ≈ 2*T*sigma_T
    df["sigma_T2"] = 2.0 * df["T_s"] * df["sigma_T_s"]
    # sigma(a^3) ≈ 3*a^2*sigma_a
    df["sigma_a3"] = 3.0 * (df["a_m"] ** 2) * df["sigma_a_m"]
    # sigma(T^2 M) via product: sqrt((M*sigma_T2)^2 + (T^2*sigma_M)^2)
    df["sigma_T2M"] = np.sqrt(
        (df["M_kg"] * df["sigma_T2"]) ** 2 + (df["T2"] * df["sigma_M_kg"]) ** 2
    )

    # Logs (for regression); filter positive values to avoid invalid logs
    valid = (df["T_s"] > 0) & (df["a_m"] > 0)
    df = df.loc[valid].reset_index(drop=True)
    df["logT"] = np.log10(df["T_s"])
    df["loga"] = np.log10(df["a_m"])

    return df

# -------------------------- Regressions ------------------------- #
def linreg(x: np.ndarray, y: np.ndarray) -> FitResult:
    res = stats.linregress(x, y)
    return FitResult(res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr)

def fit_through_origin(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Least squares fit y = m x through origin; returns (m, rmse)."""
    m = float(np.dot(x, y) / np.dot(x, x))
    yhat = m * x
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return m, rmse

# ---------------------------- Plots ----------------------------- #
def plot_logT_loga(df: pd.DataFrame, out_png: str):
    plt.figure()

    if "sigma_T_s" in df and "sigma_a_m" in df:
        # Error bars in log space need propagation:
        x = df["loga"].values
        y = df["logT"].values
        xerr = (df["sigma_a_m"] / (df["a_m"] * np.log(10))).values
        yerr = (df["sigma_T_s"] / (df["T_s"] * np.log(10))).values

        plt.errorbar(x, y, xerr=xerr, yerr=yerr,
                     fmt="o", markersize=3, ecolor="gray", alpha=0.6, capsize=2)
    else:
        plt.scatter(df["loga"], df["logT"], s=12)

    # Regression
    fr = linreg(df["loga"].values, df["logT"].values)
    x_line = np.linspace(df["loga"].min(), df["loga"].max(), 200)
    y_line = fr.slope * x_line + fr.intercept
    plt.plot(x_line, y_line, linewidth=2, color="red", label=f"Slope = {fr.slope:.2f}")
    plt.xlabel("log10(a) [m]")
    plt.ylabel("log10(T) [s]")
    plt.title("log T vs log a (expect slope ≈ 1.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return fr


def plot_T2M_vs_a3(df: pd.DataFrame, out_png: str) -> Tuple[float, float]:
    plt.figure()

    if "sigma_T2M" in df and "sigma_a3" in df:
        plt.errorbar(df["a3"], df["T2M"],
                     xerr=df["sigma_a3"], yerr=df["sigma_T2M"],
                     fmt="o", markersize=3, ecolor="gray", alpha=0.6, capsize=2)
    else:
        plt.scatter(df["a3"], df["T2M"], s=12)

    m, rmse = fit_through_origin(df["a3"].values, df["T2M"].values)
    x_line = np.linspace(df["a3"].min(), df["a3"].max(), 200)
    y_line = m * x_line
    plt.plot(x_line, y_line, linewidth=2, color="red", label=f"Slope = {m:.2e}")
    plt.xlabel("a³ [m³]")
    plt.ylabel("T² M [s²·kg]")
    plt.title("T²M vs a³ (expect slope ≈ 4π²/G)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return m, rmse

def plot_residuals_loga(df: pd.DataFrame, slope: float, intercept: float, out_png: str):
    yhat = slope * df["loga"] + intercept
    resid = df["logT"] - yhat
    plt.figure()
    plt.scatter(df["loga"], resid, s=12)
    plt.axhline(0, linewidth=1)
    plt.xlabel("log10(a)  [m]")
    plt.ylabel("Residuals in log10(T)")
    plt.title("Residuals: log T vs log a")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    
def plot_residuals_T2M_a3(df: pd.DataFrame, m_origin: float, out_png: str):
    """Residuals of T^2 M vs a^3 for fit through origin (resid = T2M - m_origin * a3)."""
    yhat = m_origin * df["a3"]
    resid = df["T2M"] - yhat
    plt.figure()
    # Use simple scatter plot (no error bars)
    plt.scatter(df["a3"], resid, s=12)
    plt.axhline(0, linewidth=1)
    plt.xlabel("a³ [m³]")
    plt.ylabel("Residuals in T²M [s²·kg]")
    plt.title("Residuals: T²M vs a³ (fit through origin)")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# --------------------------- Reporting -------------------------- #
def write_report(path: str, n_raw: int, n_clean: int, fr: FitResult, m_origin: float, rmse: float):
    lines = []
    lines.append("Kepler Exoplanet Analysis — Summary")
    lines.append("=" * 50)
    lines.append(f"Raw rows fetched: {n_raw}")
    lines.append(f"Rows after cleaning: {n_clean}")
    lines.append("")
    lines.append("Model 1: log10(T) vs log10(a)")
    lines.append(f"  Slope            : {fr.slope:.5f}  (theory ≈ {EXPECTED_SLOPE_LOG:.3f})")
    lines.append(f"  Intercept        : {fr.intercept:.5f}")
    lines.append(f"  R                : {fr.r_value:.5f}  (R^2 = {fr.r_value**2:.5f})")
    lines.append(f"  Std. Err (slope) : {fr.stderr:.5f}")
    lines.append("")
    lines.append("Model 2: T^2 M vs a^3 (fit through origin)")
    lines.append(f"  Slope (observed) : {m_origin:.6e}  SI")
    lines.append(f"  Slope (theory)   : {EXPECTED_CONST:.6e}  SI  [4π²/G]")
    lines.append(f"  RMSE             : {rmse:.6e}")
    lines.append("")
    lines.append("Interpretation notes:")
    lines.append("- Slope ≈ 1.5 in log–log supports Kepler's Third Law.")
    lines.append("- Comparing observed slope in T^2 M vs a^3 to 4π²/G tests Newtonian form,")
    lines.append("  acknowledging scatter due to measurement uncertainties and neglected effects.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ---------------------------- Main ------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Test Kepler's Third Law using NASA exoplanet data.")
    ap.add_argument("--limit", type=int, default=100, help="Number of rows to fetch from NASA (after basic WHERE filters).")
    ap.add_argument("--outdir", type=str, default="results", help="Output directory for CSV, plots, and report.")
    ap.add_argument("--require_uncertainties", action="store_true",
                    help="Drop rows without uncertainties in T, a, and M (stricter sample).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Fetching data from NASA Exoplanet Archive...")
    df_raw = fetch_dataframe(args.limit)
    n_raw = len(df_raw)
    csv_raw = outdir / "raw_ps_subset.csv"
    df_raw.to_csv(csv_raw, index=False)
    print(f"Saved raw CSV: {csv_raw}")

    print("[2/5] Cleaning, converting units, and computing derived columns...")
    df = clean_and_convert(df_raw)

    if args.require_uncertainties:
        mask = (~df["sigma_T_s"].isna()) & (~df["sigma_a_m"].isna()) & (~df["sigma_M_kg"].isna())
        df = df.loc[mask].reset_index(drop=True)

    n_clean = len(df)
    if n_clean < 5:
        print("ERROR: Too few rows after cleaning. Try increasing --limit or disabling --require_uncertainties.", file=sys.stderr)
        sys.exit(1)

    csv_clean = outdir / "cleaned_si_units.csv"
    df.to_csv(csv_clean, index=False)
    print(f"Saved cleaned CSV: {csv_clean} (rows={n_clean})")

    print("[3/5] Running regressions and generating plots...")
    # Plot 1: logT vs loga
    fr = plot_logT_loga(df, str(outdir / "logT_vs_loga.png"))
    plot_residuals_loga(df, fr.slope, fr.intercept, str(outdir / "residuals_logT_loga.png"))

    # Plot 2: T^2 M vs a^3 (fit through origin)
    m_origin, rmse = plot_T2M_vs_a3(df, str(outdir / "T2M_vs_a3.png"))
    plot_residuals_T2M_a3(df, m_origin, str(outdir / "residuals_T2M_a3.png"))

    print("[4/5] Writing summary report...")
    write_report(str(outdir / "summary_report.txt"), n_raw, n_clean, fr, m_origin, rmse)

    print("[5/5] Done.")
    print("Outputs:")
    print(f" - Raw CSV         : {csv_raw}")
    print(f" - Clean CSV       : {csv_clean}")
    print(f" - Plot (log-log)  : {outdir / 'logT_vs_loga.png'}")
    print(f" - Plot (residual) : {outdir / 'residuals_logT_loga.png'}")
    print(f" - Plot (T2M vs a3): {outdir / 'T2M_vs_a3.png'}")
    print(f" - Summary report  : {outdir / 'summary_report.txt'}")

if __name__ == "__main__":
    main()
