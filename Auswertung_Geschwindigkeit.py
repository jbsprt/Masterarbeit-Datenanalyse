import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Style-Einstellungen für wissenschaftliche Abbildungen
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# ===================================================================
# 1. KONFIGURATION & DATEIEN
# ===================================================================
SENSORS_CONFIG = {
    'Sensor_()33_30w': {'file': 'Zusammenfassung_()33_30w.xlsx'},
    'Sensor_ZP25_30w': {'file': 'Zusammenfassung_ZP25_30w.xlsx'},
}

speeds_str = [
    '0,5mm/s',
    '1mm/s',
    '1,5mm/s',
    '2mm/s',
    '2,5mm/s',
    '3mm/s',
    '3,5mm/s',
    '4mm/s',
    '4,5mm/s',
    '5mm/s',
]


def calculate_polygon_area(x, y):
  return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


all_sensors_summary = []

# ===================================================================
# 2. SCHLEIFE ÜBER SENSOREN
# ===================================================================
for sensor_name, config in SENSORS_CONFIG.items():
  try:
    df = pd.read_excel(config['file'])
  except FileNotFoundError:
    print(
        f"WARNUNG: Datei {config['file']} für {sensor_name} nicht gefunden."
        ' Wird übersprungen.'
    )
    continue

  cycle_results = []

  for sp in speeds_str:
    t_col = f'time_{sp}'
    s_col = f'strain_{sp}'
    ind_col = f'inductance_{sp}'  # Spalte für absolute Induktivität (L0)
    d_col = f'delta_I_{sp}'  # Spalte für Induktivitätsänderung (Delta L)

    if t_col not in df.columns or ind_col not in df.columns:
      continue

    sub = df[[t_col, s_col, ind_col, d_col]].dropna()
    t_vals = sub[t_col].values
    s_vals = sub[s_col].values
    ind_vals = sub[ind_col].values
    d_vals = sub[d_col].values

    speed_num = float(sp.replace('mm/s', '').replace(',', '.'))

    # ---------------------------------------------------------------
    # KORRIGIERTE L0-BERECHNUNG:
    # L0 wird aus der Spalte 'inductance_...' gelesen (~0.280 µH)
    # ---------------------------------------------------------------
    l0_mask = (t_vals <= 3.0) & (s_vals < 0.05)
    if np.sum(l0_mask) >= 5:
      l0_speed = np.mean(ind_vals[l0_mask])
    else:
      l0_speed = np.mean(ind_vals[t_vals <= 3.0])

    valleys, _ = find_peaks(-s_vals, distance=30, height=-0.2)

    # Rauschauswertung in der Ruhephase
    zero_mask = (t_vals <= 2.0) & (s_vals < 0.05)
    if np.sum(zero_mask) < 5:
      zero_mask = s_vals < 0.05
    noise_std = d_vals[zero_mask].std() if np.sum(zero_mask) > 1 else np.nan

    for i in range(len(valleys) - 1):
      s_cyc = s_vals[valleys[i] : valleys[i + 1] + 1]
      d_cyc = d_vals[valleys[i] : valleys[i + 1] + 1]

      area = calculate_polygon_area(s_cyc, d_cyc)

      # Steigung m in µH/mm
      p = np.polyfit(s_cyc, d_cyc, 1)
      slope_uH_mm = p[0]

      # Relative Sensitivität %/mm
      sensitivity_pct_mm = abs(slope_uH_mm / l0_speed) * 100.0

      # Linearität R²
      pred = np.polyval(p, s_cyc)
      ss_res = np.sum((d_cyc - pred) ** 2)
      ss_tot = np.sum((d_cyc - np.mean(d_cyc)) ** 2)
      r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

      # SNR [dB]
      amp_cyc = np.ptp(d_cyc)
      snr_db = (
          20 * np.log10(amp_cyc / noise_std)
          if (noise_std is not np.nan and noise_std > 0)
          else np.nan
      )

      cycle_results.append({
          'Sensor': sensor_name,
          'Speed_num': speed_num,
          'Cycle': i + 1,
          'L0_uH': l0_speed,
          'Sensitivity_pct_mm': sensitivity_pct_mm,
          'Area_uH_mm': area,
          'R2': r2,
          'SNR_dB': snr_db,
      })

  df_cycles = pd.DataFrame(cycle_results)

  # Statistische Zusammenfassung pro Geschwindigkeit
  metrics = ['L0_uH', 'Sensitivity_pct_mm', 'Area_uH_mm', 'SNR_dB', 'R2']
  avg = df_cycles.groupby('Speed_num')[metrics].mean()
  std = df_cycles.groupby('Speed_num')[metrics].std()

  sensor_res = pd.DataFrame({'Geschwindigkeit_mmss': avg.index})
  sensor_res['Sensor'] = sensor_name

  for m in metrics:
    sensor_res[f'{m}_Mean'] = avg[m].values
    sensor_res[f'{m}_Std'] = std[m].values
    sensor_res[f'{m}_Formatted'] = [
        f'{mean:.4f} ± {s:.4f}' if 'L0' in m else f'{mean:.3f} ± {s:.3f}'
        for mean, s in zip(avg[m], std[m])
    ]

  all_sensors_summary.append(sensor_res)

  # ===================================================================
  # PLOT PRO SENSOR (2x2 SVG Export)
  # ===================================================================
  fig, axes = plt.subplots(2, 2, figsize=(11, 9))
  plot_configs = [
      (
          axes[0, 0],
          'Sensitivity_pct_mm',
          'Sensitivität [%/mm]',
          'Relative Sensitivität',
          '#1f77b4',
      ),
      (
          axes[0, 1],
          'Area_uH_mm',
          'Hysteresefläche [µH·mm]',
          'Hysteresefläche',
          '#d62728',
      ),
      (axes[1, 0], 'SNR_dB', 'SNR [dB]', 'Signal-Rausch-Verhältnis (SNR)', '#2ca02c'),
      (
          axes[1, 1],
          'R2',
          r'Bestimmtheitsmaß $R^2$ [-]',
          'Linearität ($R^2$)',
          '#ff7f0e',
      ),
  ]

  for ax, m, ylabel, title, color in plot_configs:
    ax.errorbar(
        avg.index,
        avg[m],
        yerr=std[m],
        fmt='-o',
        color=color,
        ecolor='#333333',
        capsize=4,
        markersize=6,
        linewidth=1.5,
    )
    ax.set_xlabel('Geschwindigkeit [mm/s]', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xticks(avg.index)
    ax.grid(True, linestyle=':', alpha=0.6)

  plt.suptitle(
      f'Geschwindigkeitseinfluss: {sensor_name}',
      fontsize=12,
      fontweight='bold',
  )
  plt.tight_layout()
  plt.savefig(f'fig_geschwindigkeit_{sensor_name}.svg', format='svg')
  plt.close()

# ===================================================================
# 3. ZUSAMMENFASSENDE EXCEL ERSTELLEN
# ===================================================================
if all_sensors_summary:
  df_final_summary = pd.concat(all_sensors_summary, ignore_index=True)
  df_final_summary.to_excel(
      'Zusammenfassung_Sensoren_Vergleich.xlsx', index=False
  )
  print(
      'Auswertung abgeschlossen! Ergebnisse in'
      " 'Zusammenfassung_Sensoren_Vergleich.xlsx' gespeichert."
  )