import argparse
import io
from pathlib import Path
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Constants (SI Units) ---
G = 6.67430e-11           # Gravitational Constant 
M_SUN = 1.98847e30        # Solar Mass
AU = 1.495978707e11       # Astronomical Unit
DAY = 86400.0             # Day

def fetch_data(limit, unique):
    cols = [
        "pl_name", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2",
        "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2",
        "st_mass", "st_masserr1", "st_masserr2"
    ]
    table = 'pscomppars' if unique else 'ps'
    query = f"SELECT TOP {limit} {','.join(cols)} FROM {table} " \
            f"WHERE pl_orbper IS NOT NULL AND pl_orbsmax IS NOT NULL AND st_mass IS NOT NULL"
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={requests.utils.quote(query)}&format=csv"
    
    print(f"Fetching {limit} records from NASA...")
    r = requests.get(url)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def clean_and_convert(df, outdir):
    # 1. Save Raw Data
    df.to_csv(outdir / "1_raw_data_nasa.csv", index=False)
    
    # 2. Ensure absolute values for errors
    err_cols = ['pl_orbpererr2', 'pl_orbsmaxerr2', 'st_masserr2']
    for col in err_cols:
        df[col] = df[col].abs()
        
    # 3. Filter: Relative Uncertainty < 20%
    max_err_T = df[['pl_orbpererr1', 'pl_orbpererr2']].max(axis=1)
    max_err_a = df[['pl_orbsmaxerr1', 'pl_orbsmaxerr2']].max(axis=1)
    max_err_M = df[['st_masserr1', 'st_masserr2']].max(axis=1)
    
    mask = (max_err_T / df['pl_orbper'] <= 0.2) & \
           (max_err_a / df['pl_orbsmax'] <= 0.2) & \
           (max_err_M / df['st_mass'] <= 0.2)
    df_clean = df[mask].copy()
    
    # 4. Convert to SI Units
    df_clean['T_s'] = df_clean['pl_orbper'] * DAY
    df_clean['T_s_err1'] = df_clean['pl_orbpererr1'] * DAY
    df_clean['T_s_err2'] = df_clean['pl_orbpererr2'] * DAY
    
    df_clean['a_m'] = df_clean['pl_orbsmax'] * AU
    df_clean['a_m_err1'] = df_clean['pl_orbsmaxerr1'] * AU
    df_clean['a_m_err2'] = df_clean['pl_orbsmaxerr2'] * AU
    
    df_clean['M_kg'] = df_clean['st_mass'] * M_SUN
    df_clean['M_kg_err1'] = df_clean['st_masserr1'] * M_SUN
    df_clean['M_kg_err2'] = df_clean['st_masserr2'] * M_SUN

    df_clean.to_csv(outdir / "2_clean_data_si.csv", index=False)
    return df_clean

def analyze_kepler_original(df, outdir):
    print("\n--- Graph 1: Kepler's Original Law ---")
    
    # Linearize: log(T) vs log(a)
    x = np.log10(df['a_m'])
    y = np.log10(df['T_s'])
    
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
    r_squared = r_val**2

    # Save plotting data
    plot_df = pd.DataFrame({'log_a': x, 'log_T': y, 'a_m': df['a_m'], 'T_s': df['T_s']})
    plot_df.to_csv(outdir / "3_plot_data_graph1.csv", index=False)

    plt.figure(figsize=(10, 7))
    plt.errorbar(df['a_m'], df['T_s'], 
                 xerr=[df['a_m_err2'], df['a_m_err1']], 
                 yerr=[df['T_s_err2'], df['T_s_err1']],
                 fmt='o', markersize=3, color='#7f8c8d', alpha=0.3, label='Data')
    
    x_range = np.logspace(np.log10(df['a_m'].min()), np.log10(df['a_m'].max()), 100)
    y_fit = (10**intercept) * (x_range**slope)
    plt.plot(x_range, y_fit, color='#e67e22', linewidth=2, label=f'Slope={slope:.3f}, $R^2$={r_squared:.4f}')

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Semi-major Axis (m)')
    plt.ylabel('Orbital Period (s)')
    plt.title("Graph 1: Kepler's Original Law (T vs a)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    plt.savefig(outdir / "graph1_kepler_original.png", dpi=200)
    print(f"R^2: {r_squared:.5f}")

def analyze_newton_modified(df, outdir):
    print("\n--- Graph 2: Newton's Modified Law ---")
    
    # Calculate Y = T * sqrt(M)
    df['Y_val'] = df['T_s'] * np.sqrt(df['M_kg'])
    
    # Error Propagation (Quadrature)
    def calc_y_err(t_err, m_err):
        return df['Y_val'] * np.sqrt( (t_err/df['T_s'])**2 + (0.5 * m_err/df['M_kg'])**2 )
    
    df['Y_err_hi'] = calc_y_err(df['T_s_err1'], df['M_kg_err1'])
    df['Y_err_lo'] = calc_y_err(df['T_s_err2'], df['M_kg_err2'])

    # Linearize: log(Y) vs log(a)
    x = np.log10(df['a_m'])
    y = np.log10(df['Y_val'])
    
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
    r_squared = r_val**2

    # Theoretical Intercept: log(2π/sqrt(G))
    theo_intercept = np.log10( (2 * np.pi) / np.sqrt(G) )
    pct_error = abs((intercept - theo_intercept) / theo_intercept) * 100

    # Save plotting data
    plot_df = pd.DataFrame({'log_a': x, 'log_Y': y, 'a_m': df['a_m'], 'Y_val': df['Y_val']})
    plot_df.to_csv(outdir / "4_plot_data_graph2.csv", index=False)

    plt.figure(figsize=(10, 7))
    plt.errorbar(df['a_m'], df['Y_val'], 
                 xerr=[df['a_m_err2'], df['a_m_err1']], 
                 yerr=[df['Y_err_lo'], df['Y_err_hi']],
                 fmt='o', markersize=3, color='#2c3e50', alpha=0.3, label='Data (SI)')
    
    x_range = np.logspace(np.log10(df['a_m'].min()), np.log10(df['a_m'].max()), 100)
    y_fit = (10**intercept) * (x_range**slope)
    plt.plot(x_range, y_fit, color='#e74c3c', linewidth=2, label=f'Slope={slope:.3f}, $R^2$={r_squared:.4f}')

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Semi-major Axis (m)')
    plt.ylabel(r'Normalized Period ($T \sqrt{M}$) [s $\cdot$ kg$^{1/2}$]')
    plt.title("Graph 2: Newton's Modified Law (Mass Corrected)")
    
    # Add stats to plot
    stats_box = (f"Slope: {slope:.4f} (Theory: 1.5)\n"
                 f"Intercept: {intercept:.4f} (Theory: {theo_intercept:.4f})\n"
                 f"R²: {r_squared:.5f}\n"
                 f"Intercept Error: {pct_error:.2f}%")
    plt.gca().text(0.05, 0.8, stats_box, transform=plt.gca().transAxes, 
                   fontsize=9, bbox=dict(facecolor='white', alpha=0.9))
    
    plt.legend(loc='lower right')
    plt.grid(True, which="both", alpha=0.2)
    plt.savefig(outdir / "graph2_newton_modified.png", dpi=200)
    print(f"R^2: {r_squared:.5f}")
    print(f"Intercept % Error: {pct_error:.2f}%")

if __name__ == "__main__":
    outdir = Path("results_ia2"); outdir.mkdir(parents=True, exist_ok=True)
    df = clean_and_convert(fetch_data(6000, True), outdir)
    analyze_kepler_original(df, outdir)
    analyze_newton_modified(df, outdir)
