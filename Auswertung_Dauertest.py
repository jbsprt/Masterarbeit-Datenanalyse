import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

# ===================================================================
# EINSTELLUNGEN & DATEINAMEN
# ===================================================================
INPUT_FILE = "dauerlauf_)(33_30w.csv"
OUTPUT_CSV = "zyklen_gemittelt_)(33_30w.csv"
OUTPUT_PLOT = "dauerlauf_)(33_30w.png"

# Mittelungs-Intervall (Zyklen pro Block)
BIN_SIZE = 50

print("Lade gefilterte Messdaten...")
df = pd.read_csv(INPUT_FILE, sep=";", decimal=",")

SIGNAL_COL = "inductance_uH"

# ===================================================================
# 1. ZYKLEN-SEGMENTIERUNG & KENNWERTERMITTLUNG
# ===================================================================
print("Segmentiere Zyklen anhand der Dehnungs-Täler...")
valleys, _ = find_peaks(-df["strain_mm"], distance=80, height=-0.2)

print(f"{len(valleys) - 1} Zyklen erkannt.")

results = []

for i in range(len(valleys) - 1):
    # Ersten Zyklus (i == 0) überspringen
    if i == 0:
        continue

    idx_start = valleys[i]
    idx_end = valleys[i+1]
    
    c_df = df.iloc[idx_start:idx_end].copy().reset_index(drop=True)
    
    if len(c_df) < 10:
        continue

    t_start = c_df["time_s"].iloc[0]
    
    strain = c_df["strain_mm"].values
    signal = c_df[SIGNAL_COL].values
    
    idx_strain_min = np.argmin(strain)
    idx_strain_max = np.argmax(strain)
    
    nullpunkt = signal[idx_strain_min]
    peak_val = signal[idx_strain_max]
    amplitude = abs(peak_val - nullpunkt)
    
    delta_strain = strain[idx_strain_max] - strain[idx_strain_min]
    sensitivity = amplitude / delta_strain if delta_strain > 0 else np.nan
    
    # Hysteresefläche
    strain_up = strain[:idx_strain_max + 1]
    sig_up = signal[:idx_strain_max + 1]
    strain_down = strain[idx_strain_max:]
    sig_down = signal[idx_strain_max:]
    
    area_up = trapezoid(sig_up, strain_up)
    area_down = trapezoid(sig_down, strain_down)
    hysteresis_area = abs(area_up + area_down)
    
    # Bestimmtheitsmaß R² (Anstelle der Linearitätsabweichung % FS)
    if len(strain_up) > 2 and delta_strain > 0:
        poly = np.polyfit(strain_up, sig_up, 1)
        sig_fit = np.polyval(poly, strain_up)
        
        ss_res = np.sum((sig_up - sig_fit) ** 2)
        ss_tot = np.sum((sig_up - np.mean(sig_up)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    else:
        r2_score = np.nan
        
    signal_std_noise = np.std(sig_up[:5]) if len(sig_up) >= 5 else np.nan
    
    results.append({
        "cycle": i + 1,
        "time_s": t_start,
        "nullpunkt_uH": nullpunkt,
        "peak_uH": peak_val,
        "amplitude_uH": amplitude,
        "sensitivity_uH_mm": sensitivity,
        "hysteresis_area_uH_mm": hysteresis_area,
        "r2_score": r2_score,
        "noise_std_uH": signal_std_noise
    })

res_df = pd.DataFrame(results)

# ===================================================================
# 2. MITTELUNG ÜBER JEWEILS 50 ZYKLEN
# ===================================================================
print(f"Mittele Daten über Blöcke von {BIN_SIZE} Zyklen...")

# Gruppierungs-Index (Zyklus 2..51 -> Gruppe 1, 52..101 -> Gruppe 2, ...)
res_df["cycle_group"] = ((res_df["cycle"] - 2) // BIN_SIZE) + 1

# Aggregation durchführen
grouped_df = res_df.groupby("cycle_group").agg({
    "cycle": ["min", "max", "mean"],
    "time_s": "mean",
    "nullpunkt_uH": "mean",
    "peak_uH": "mean",
    "amplitude_uH": "mean",
    "sensitivity_uH_mm": "mean",
    "hysteresis_area_uH_mm": "mean",
    "r2_score": "mean",
    "noise_std_uH": "mean"
}).reset_index()

# Spaltennamen definieren
grouped_df.columns = [
    "group_id", "cycle_start", "cycle_end", "cycle_mean", "time_s_mean",
    "nullpunkt_uH_mean", "peak_uH_mean", "amplitude_uH_mean",
    "sensitivity_uH_mm_mean", "hysteresis_area_uH_mm_mean",
    "r2_score_mean", "noise_std_uH_mean"
]

# Speichern der gemittelten Ergebnisse
grouped_df.to_csv(OUTPUT_CSV, sep=";", decimal=",", index=False)
print(f"Gemittelte Ergebnisse gespeichert in: {OUTPUT_CSV}")

# ===================================================================
# 3. VISUALISIERUNG DER GEMITTELTEN TRENDS
# ===================================================================
print("Erstelle Auswertediagramme für gemittelte Daten...")
fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
fig.suptitle(f"Dauerlauf-Auswertung (Gemittelt über je {BIN_SIZE} Zyklen, ohne Zyklus 1)", fontsize=16)

x_axis = grouped_df["cycle_mean"]

# Graph 1: Nullpunktdrift & Peakdrift
axs[0, 0].plot(x_axis, grouped_df["nullpunkt_uH_mean"], label="Nullpunkt (0 mm)", color="blue")
axs[0, 0].plot(x_axis, grouped_df["peak_uH_mean"], label="Peak (2 mm)", color="red", alpha=0.7)
axs[0, 0].set_ylabel("Induktivität [µH]")
axs[0, 0].set_title("1. Nullpunkt- & Maximumsdrift (Gemittelt)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Graph 2: Amplitudenverlauf
axs[0, 1].plot(x_axis, grouped_df["amplitude_uH_mean"], color="green")
axs[0, 1].set_ylabel("Amplitude ΔL [µH]")
axs[0, 1].set_title("2. Amplitudenänderung (Gemittelt)")
axs[0, 1].grid(True)

# Graph 3: Hysteresefläche
axs[1, 0].plot(x_axis, grouped_df["hysteresis_area_uH_mm_mean"], color="purple")
axs[1, 0].set_ylabel("Fläche [µH·mm]")
axs[1, 0].set_title("3. Hysteresefläche (Gemittelt)")
axs[1, 0].grid(True)

# Graph 4: Sensitivität
axs[1, 1].plot(x_axis, grouped_df["sensitivity_uH_mm_mean"], color="orange")
axs[1, 1].set_ylabel("Sensitivität [µH/mm]")
axs[1, 1].set_title("4. Empfindlichkeit (Gemittelt)")
axs[1, 1].grid(True)
axs[1, 1].tick_params(labelbottom=True)
axs[1, 1].set_xlabel("Zyklus (Mittelwert)")

# Graph 5: Bestimmtheitsmaß R²
axs[2, 0].plot(x_axis, grouped_df["r2_score_mean"], color="darkred")
axs[2, 0].set_xlabel("Zyklus (Mittelwert)")
axs[2, 0].set_ylabel("R²-Wert")
axs[2, 0].set_title("5. Linearität (Bestimmtheitsmaß R²)")
axs[2, 0].grid(True)

# Das 6. (leere) Subplot-Fenster entfernen
fig.delaxes(axs[2, 1])

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
print(f"Diagramm gespeichert unter: {OUTPUT_PLOT}")