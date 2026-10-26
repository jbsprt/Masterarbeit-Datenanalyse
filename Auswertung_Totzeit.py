import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ===================================================================
# 0. GLOBALE SCHRIFTGRÖSSEN EINSTELLEN
# ===================================================================
plt.rcParams.update({
    'font.size': 12,           # Standard-Schriftgröße
    'axes.labelsize': 14,      # Achsenbeschriftung (xlabel, ylabel)
    'axes.titlesize': 15,      # Subplot-Titel
    'xtick.labelsize': 12,     # Werte auf der X-Achse
    'ytick.labelsize': 12,     # Werte auf der Y-Achse
    'legend.fontsize': 11,     # Schriftgröße in der Legende
    'figure.titlesize': 18     # Gesamt-Titel
})

# ===================================================================
# 1. DATEN EINLESEN
# ===================================================================
file_path = "Totzeittest_2mms.xlsx"
sheet_name = "Tabelle1"
df = pd.read_excel(file_path, sheet_name=sheet_name)

sensors = ["()33_30w", ")(33_30w"]
amplitudes = ["1mm", "1,5mm", "2mm"]

results_list = []

# ===================================================================
# 2. ZEITVERLÄUFE PLOTTEN (TOTZEIT BEIM VERLASSEN DES PLATEAUS)
# ===================================================================
for sensor in sensors:
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=False)
    fig.suptitle(
        f"Sensor: {sensor}",
        fontsize=18,
        fontweight="bold",
    )

    for i, amp in enumerate(amplitudes):
        ds_name = f"{sensor}_{amp}"
        t_col = f"time_{ds_name}"
        s_col = f"strain_{ds_name}"
        di_col = f"delta_I_{ds_name}"

        if t_col not in df.columns:
            continue

        data = df[[t_col, s_col, di_col]].dropna().reset_index(drop=True)
        t = data[t_col].values
        strain = data[s_col].values
        delta_i = data[di_col].values

        # Ableitung (Geschwindigkeit) der Dehnung berechnen
        dt = np.diff(t)
        dstrain = np.diff(strain) / np.where(dt == 0, 1e-6, dt)
        dstrain = np.append(dstrain, 0)

        # Grundrauschen der Induktivität im Ruhezustand ermitteln
        baseline_noise_std = np.std(delta_i[:500])
        threshold_3sigma = 3 * baseline_noise_std

        # Zyklen-Startpunkte finden (Täler der Dehnung)
        minima_idx, _ = find_peaks(-strain, distance=100, prominence=0.1)
        if len(minima_idx) == 0 or minima_idx[0] != 0:
            minima_idx = np.insert(minima_idx, 0, 0)
        if minima_idx[-1] != len(strain) - 1:
            minima_idx = np.append(minima_idx, len(strain) - 1)

        ax_left = axes[i]
        ax_right = ax_left.twinx()

        line1 = ax_left.plot(
            t, delta_i, color="#1f77b4", label=r"$\Delta L$ (µH)", linewidth=1.2
        )
        line2 = ax_right.plot(
            t,
            strain,
            color="#d62728",
            linestyle="--",
            label="Dehnung (mm)",
            linewidth=1.2,
        )
        ax_right.invert_yaxis()

        first_red_label, first_green_label = True, True
        cycle_dead_times = []

        for c in range(len(minima_idx) - 1):
            c_start = minima_idx[c]
            c_end = minima_idx[c + 1]

            c_strain = strain[c_start:c_end]
            c_dstrain = dstrain[c_start:c_end]
            c_delta_i = delta_i[c_start:c_end]
            c_t = t[c_start:c_end]

            # Maximum der Dehnung (Start des Plateau-Zustands) finden
            peak_idx = np.argmax(c_strain)

            # MECHANISCHER START DER ENTLASTUNG:
            unloading_start_rel = np.where(
                (np.arange(len(c_dstrain)) >= peak_idx) & (c_dstrain < -0.5)
            )[0]
            if len(unloading_start_rel) == 0:
                continue

            c_mech_idx = unloading_start_rel[0]
            t_mech = c_t[c_mech_idx]

            # SIGNAL-REFERENZ: Mittelwert der letzten 100 ms des Plateaus vor t_mech
            plateau_win_indices = np.where((c_t >= t_mech - 0.100) & (c_t <= t_mech))[0]
            if len(plateau_win_indices) > 0:
                di_baseline_cycle = np.mean(c_delta_i[plateau_win_indices])
            else:
                di_baseline_cycle = c_delta_i[c_mech_idx]

            # ELEKTRISCHER START: Abweichung vom Plateau-Mittelwert um > 3-Sigma
            post_start_indices = np.where(c_t >= t_mech)[0]
            elec_dev = np.abs(c_delta_i[post_start_indices] - di_baseline_cycle)
            above_thresh = np.where(elec_dev > threshold_3sigma)[0]

            if len(above_thresh) > 0:
                t_elec = c_t[post_start_indices[above_thresh[0]]]
            else:
                t_elec = t_mech

            dead_time_ms = (t_elec - t_mech) * 1000.0
            cycle_dead_times.append(dead_time_ms)

            # Vertikale Linien zeichnen
            ax_left.axvline(
                x=t_mech,
                color="red",
                linestyle="--",
                linewidth=0.8,
                alpha=0.8,
                label="Beginn Entlastung" if first_red_label else "",
            )
            first_red_label = False

            ax_left.axvline(
                x=t_elec,
                color="green",
                linestyle=":",
                linewidth=1.0,
                alpha=0.9,
                label="3σ Schwelle" if first_green_label else "",
            )
            first_green_label = False

        # Mittelwert und Standardabweichung der Totzeit
        avg_dead_time_ms = (
            np.mean(cycle_dead_times) if len(cycle_dead_times) > 0 else 0.0
        )
        std_dead_time_ms = (
            np.std(cycle_dead_times, ddof=1) if len(cycle_dead_times) > 1 else 0.0
        )

        # Viskoelastische Relaxation im 1. Zyklus
        max_strain = np.max(strain)
        hold_indices = np.where(strain >= 0.98 * max_strain)[0]

        if len(hold_indices) > 0:
            diffs = np.diff(hold_indices)
            first_break = np.where(diffs > 1)[0]
            cycle1_hold_idx = (
                hold_indices[: first_break[0] + 1]
                if len(first_break) > 0
                else hold_indices
            )

            di_hold_start = delta_i[cycle1_hold_idx[0]]
            di_hold_end = delta_i[cycle1_hold_idx[-1]]

            relaxation_drift_uH = di_hold_end - di_hold_start
            relaxation_drift_pct = (
                abs((relaxation_drift_uH / di_hold_start) * 100.0)
                if di_hold_start != 0
                else 0.0
            )
        else:
            relaxation_drift_uH = np.nan
            relaxation_drift_pct = np.nan

        results_list.append({
            "Datensatz": ds_name,
            "Max_Dehnung_mm": max_strain,
            "Totzeit_Mean_ms": round(avg_dead_time_ms, 1),
            "Totzeit_Std_ms": round(std_dead_time_ms, 1),
            "Relaxation_Drift_uH": relaxation_drift_uH,
            "Relaxation_Drift_Prozent": relaxation_drift_pct,
        })

        # --- VERGRÖSSERTE BESCHRIFTUNGEN UND SKALEN ---
        ax_left.set_title(
            f"Längenänderung: {amp} | Totzeit: {avg_dead_time_ms:.1f} ± {std_dead_time_ms:.1f} ms",
            fontsize=15,
            fontweight="bold",
            pad=10
        )
        ax_left.set_xlabel("Zeit [s]", fontsize=14, fontweight="bold")
        ax_left.set_ylabel(r"$\Delta L$ [µH]", color="#1f77b4", fontsize=14, fontweight="bold")
        ax_right.set_ylabel("Längenänderung [mm]", color="#d62728", fontsize=14, fontweight="bold")

        # Achsenskalen (Zahlenwerte) vergrößern
        ax_left.tick_params(axis="both", which="major", labelsize=12)
        ax_right.tick_params(axis="both", which="major", labelsize=12)

        ax_left.grid(True, linestyle=":", alpha=0.6)

        # Legende mit vergrößerter Schrift
        lines, labels = ax_left.get_legend_handles_labels()
        lines_r, labels_r = ax_right.get_legend_handles_labels()
        ax_left.legend(
            lines + lines_r, labels + labels_r, loc="lower left", fontsize=11, ncol=2
        )

    fig.tight_layout()
    plt.show()

# ===================================================================
# 3. EXPORT DER KENNWERTE ALS EXCEL-DATEI
# ===================================================================
df_results = pd.DataFrame(results_list)
output_excel_path = "Auswertung_Totzeit_Plateauende.xlsx"
df_results.to_excel(output_excel_path, index=False)

print(f"Auswertung erfolgreich als Excel gespeichert unter: {output_excel_path}")
print(df_results)