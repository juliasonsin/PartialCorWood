import sys
import traceback

try:
    import subprocess

    # --- AUTOMATIC PACKAGE INSTALLER ---
    def check_and_install(package, import_name=None):
        try: 
            __import__(import_name or package)
        except ImportError: 
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    print("[INFO] Checking system dependencies...")
    check_and_install('pandas')
    check_and_install('numpy')
    check_and_install('scipy')
    check_and_install('scikit-bio', 'skbio')
    check_and_install('matplotlib', 'matplotlib')
    check_and_install('seaborn')

    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    from skbio.stats.composition import clr
    from sklearn.preprocessing import StandardScaler

    print("\nHINT: Ensure you type the file name WITH the extension (e.g., data.csv or data.txt)")
    path = input("\nEnter the file name or path: ").strip().replace("'", "").replace('"', "")
    
    # --- 1. UNIVERSAL FILE READER ---
    print("\n[INFO] Reading file...")
    try:
        df = pd.read_csv(path, sep=';', decimal=',')
        if df.shape[1] < 2:
            df = pd.read_csv(path, sep='\t', decimal=',')
        if df.shape[1] < 2:
            df = pd.read_csv(path, sep=',', decimal='.')
        if df.shape[1] < 2:
            raise ValueError("Data separator not recognized. Ensure it is a valid CSV or TXT.")
    except Exception as e:
        raise ValueError(f"Could not read the file. Check if the name is correct. Detail: {e}")

    # --- 2. DEEP CLEANING ---
    df.columns = df.columns.str.strip()
    
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.')
    
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='any')
    
    print(f"[INFO] Valid samples (rows) found for analysis: {len(df)}")
    if len(df) <= 3:
        raise ValueError("The model needs at least 4 valid rows to calculate Partial Correlation.")

    print("\nAvailable columns:", ", ".join(df.columns))
    y_target = input("\nType the exact name of your TARGET variable (e.g., VS): ").strip()
    z_control = input("Type the exact name of your CONTROL variable (e.g., WD): ").strip()

    if y_target not in df.columns or z_control not in df.columns:
        raise ValueError(f"Columns '{y_target}' or '{z_control}' not found in the table.")

    X_raw = df.drop(columns=[y_target, z_control]).copy()

    # --- 3. SUB-COMPOSITIONAL CLR TRANSFORMATION ---
    vessel_cols = [c for c in X_raw.columns if c in ['SVP_%', 'MVP_%']]
    tissue_cols = [c for c in X_raw.columns if c in ['VLuP_%', 'VWP_%', 'FLuP_%', 'FWP_%', 'RP_%', 'APP_%']]

    if vessel_cols: 
        print("[INFO] Applying CLR on Vessel sub-composition.")
        X_raw[vessel_cols] = clr(X_raw[vessel_cols].replace(0, 1e-6))
    if tissue_cols: 
        print("[INFO] Applying CLR on Tissue sub-composition.")
        X_raw[tissue_cols] = clr(X_raw[tissue_cols].replace(0, 1e-6))
    
    cols_to_drop = [c for c in ['SVP_%', 'VWP_%'] if c in X_raw.columns]
    if cols_to_drop:
        X_raw = X_raw.drop(columns=cols_to_drop)
        print(f"[INFO] Removed baselines {cols_to_drop} after CLR.")

    # --- 4. STANDARDIZATION ---
    scaler = StandardScaler()
    X_sc_df = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)
    
    df_final = X_sc_df.copy()
    df_final.reset_index(drop=True, inplace=True)
    df_final[y_target] = df[y_target].reset_index(drop=True)
    df_final[z_control] = df[z_control].reset_index(drop=True)

    # --- 5. EXACT PARTIAL CORRELATION ANALYSIS (Mathematical Formula) ---
    print(f"\n[INFO] Running Partial Correlation against '{y_target}', controlling for '{z_control}'...")
    results = []
    
    for col in X_sc_df.columns:
        if df_final[col].std() == 0:
            print(f"[WARNING] Skipping '{col}' (No variance found).")
            continue
        
        try:
            # Correlações simples (Pearson)
            r_xy, _ = stats.pearsonr(df_final[col], df_final[y_target])
            r_xz, _ = stats.pearsonr(df_final[col], df_final[z_control])
            r_yz, _ = stats.pearsonr(df_final[y_target], df_final[z_control])

            # Fórmula da Correlação Parcial (1 variável de controle)
            numerator = r_xy - (r_xz * r_yz)
            denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
            
            if denominator == 0:
                continue
                
            r_val = numerator / denominator
            # Prevenção contra micro-arredondamentos do Python (ex: 1.0000000000000002)
            r_val = max(min(r_val, 1.0), -1.0)
            
            # Cálculo exato do P-valor (Graus de liberdade = N - 3)
            dof = len(df_final) - 3
            if abs(r_val) == 1.0:
                p_val = 0.0
            else:
                t_stat = r_val * np.sqrt(dof / (1 - r_val**2))
                p_val = 2 * stats.t.sf(np.abs(t_stat), dof)
            
            p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.3f}"
            sig = "Yes (*)" if p_val < 0.05 else "No"
            
            results.append({
                'Anatomical_Feature': col, 
                'Partial_R': r_val, 
                'p-value_num': p_val, 
                'p-value_str': p_str, 
                'Significant': sig
            })
        except Exception as e:
            print(f"[WARNING] Error calculating {col}: {e}")

    if not results:
        raise ValueError("No valid correlations could be calculated.")

    # --- 6. EXPORT AND PLOTTING ---
    df_results = pd.DataFrame(results).sort_values(by='Partial_R', ascending=False)
    
    export_csv = f"partial_corr_{y_target}_vs_{z_control}.csv"
    df_results.drop(columns=['p-value_num']).to_csv(export_csv, index=False, sep=';', decimal=',')

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    colors = [
        'tab:blue' if (p < 0.05 and r > 0) else 'tab:red' if (p < 0.05 and r < 0) else 'lightgrey' 
        for p, r in zip(df_results['p-value_num'], df_results['Partial_R'])
    ]

    ax = sns.barplot(data=df_results, x='Partial_R', y='Anatomical_Feature', palette=colors)
    max_r = df_results['Partial_R'].abs().max() or 1.0

    for i, (p_str, r_val, sig) in enumerate(zip(
        df_results['p-value_str'], 
        df_results['Partial_R'], 
        df_results['Significant']
    )):
        align = 'left' if r_val > 0 else 'right'
        offset = max_r * 0.02 if r_val > 0 else -max_r * 0.02
        text = f"p={p_str}" + (" *" if sig == "Yes (*)" else "")
        ax.text(r_val + offset, i, text, va='center', ha=align, fontsize=10, color='black')

    plt.axvline(0, color='black', linewidth=1.2)
    plt.xlim(-max_r * 1.3, max_r * 1.3)
    
    plt.title(f"Partial Correlation with {y_target} (Controlling for {z_control})", fontsize=14, fontweight='bold')
    plt.xlabel("Partial Correlation Coefficient (r)", fontsize=12)
    plt.ylabel("Anatomical Features (Post-CLR)", fontsize=12)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:blue', label='Significant Positive (p < 0.05)'),
        Patch(facecolor='tab:red', label='Significant Negative (p < 0.05)'),
        Patch(facecolor='lightgrey', label='Non-Significant')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    plot_name = f"partial_corr_plot_{y_target}_vs_{z_control}.png"
    plt.savefig(plot_name, dpi=300)
    plt.close()

    print(f"\n[SUCCESS] Analysis completed successfully!")
    print(f" -> Table saved: {export_csv} (Open in Excel)")
    print(f" -> High-res graph saved: {plot_name}")
    input("\nPress ENTER to close...")

# --- GLOBAL ERROR TRAP ---
except Exception as e:
    print("\n" + "="*50)
    print(" 🚨 CRITICAL ERROR ENCOUNTERED 🚨 ")
    print("="*50)
    traceback.print_exc()
    print("="*50)
    input("\nThe program paused. Take a photo or copy the text above, then press ENTER to exit...")