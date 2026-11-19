import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.stats import ttest_rel


def get_p_asterisk(p_val):
    """Gibt ein Sternchen zurück, wenn p < 0.05 ist."""
    if not np.isnan(p_val) and p_val < 0.05:
        return '*'
    return ''


def evaluate_filters_on_excel_uH(file_path):
    df = pd.read_excel(file_path)
    cols = df.columns

    all_results = []

    for i in range(0, len(cols), 5):
        t_col, s_col, i_col = cols[i], cols[i + 1], cols[i + 2]
        sensor_name = t_col.replace('time_', '').replace('time', '')

        sub = df[[t_col, s_col, i_col]].dropna()
        sub.columns = ['time', 'strain', 'inductance']

        t = sub['time'].values
        strain = sub['strain'].values
        raw_ind = sub['inductance'].values  # in µH

        dt = np.mean(np.diff(t))
        fs = 1.0 / dt

        # 1. Savitzky-Golay (symmetrisch)
        window_sg = 21
        poly_sg = 3
        ind_sg = signal.savgol_filter(
            raw_ind, window_length=window_sg, polyorder=poly_sg
        )

        # 2. Butterworth Lowpass (Zero-Phase via filtfilt)
        cutoff = 2.0  # Hz
        b_bw, a_bw = signal.butter(N=2, Wn=cutoff, btype='low', fs=fs)
        ind_bw = signal.filtfilt(b_bw, a_bw, raw_ind)

        # 3. Moving Average (Zentriert via filtfilt)
        window_ma = 21
        kernel = np.ones(window_ma) / window_ma
        ind_ma = signal.filtfilt(kernel, [1.0], raw_ind)

        filters = {
            'Rohdaten': raw_ind,
            'Savitzky-Golay': ind_sg,
            'Butterworth': ind_bw,
            'gleitender Mittelwert': ind_ma,
        }

        # Ruhephase wie in Auswertung.py (für Rohdaten-SNR)
        raw_zero_mask = (t <= 3.0) & (strain < 0.05)
        if np.sum(raw_zero_mask) < 10:
            raw_zero_mask = (t <= 3.0)
        raw_noise_std = raw_ind[raw_zero_mask].std()

        # Ruhephase wie ursprünglich in Vergleich_filter.py (für Filtervergleich)
        zero_mask = (t <= 2.0) & (strain == 0)
        if np.sum(zero_mask) < 10:
            zero_mask = (t <= 100) & (strain < 0.05)

        peaks, _ = signal.find_peaks(strain, distance=100, height=1.8)
        valleys, _ = signal.find_peaks(-strain, distance=100, height=-0.2)
        min_len = min(len(peaks), len(valleys))

        raw_amp_mean = np.mean(
            np.abs(raw_ind[peaks[:min_len]] - raw_ind[valleys[:min_len]])
        )
        raw_snr_db = (
            20 * np.log10(raw_amp_mean / raw_noise_std)
            if raw_noise_std > 0
            else np.nan
        )

        for f_name, f_ind in filters.items():
            noise_std = f_ind[zero_mask].std()

            amp_mean = np.mean(
                np.abs(f_ind[peaks[:min_len]] - f_ind[valleys[:min_len]])
            )

            if f_name == 'Rohdaten':
                snr_db = raw_snr_db
            else:
                snr_db = (
                    20 * np.log10(amp_mean / noise_std)
                    if noise_std > 0
                    else np.nan
                )

            p = np.polyfit(strain, f_ind, 1)
            pred = np.polyval(p, strain)
            ss_res = np.sum((f_ind - pred) ** 2)
            ss_tot = np.sum((f_ind - np.mean(f_ind)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

            fso = np.ptp(f_ind)
            max_dev = np.max(np.abs(f_ind - pred))
            lin_dev_pct = (max_dev / fso) * 100 if fso > 0 else np.nan

            all_results.append({
                'Sensor': sensor_name,
                'Filter': f_name,
                'Rauschen_Std_uH': noise_std,
                'Signalamplitude_uH': amp_mean,
                'SNR_dB': snr_db,
                'R2_Linearitaet': r2,
                'Lin_Abweichung_pct_FSO': lin_dev_pct,
            })

    res_df = pd.DataFrame(all_results)

    metrics = [
        'Rauschen_Std_uH',
        'Signalamplitude_uH',
        'SNR_dB',
        'R2_Linearitaet',
        'Lin_Abweichung_pct_FSO',
    ]

    filters_order = [
        'Rohdaten',
        'Savitzky-Golay',
        'Butterworth',
        'gleitender Mittelwert',
    ]

    avg_summary = res_df.groupby('Filter')[metrics].mean().reindex(filters_order)
    std_summary = res_df.groupby('Filter')[metrics].std().reindex(filters_order)

    # Paarweiser t-Test gegenüber 'Rohdaten'
    raw_df = (
        res_df[res_df['Filter'] == 'Rohdaten']
        .sort_values('Sensor')
        .reset_index(drop=True)
    )
    p_values = {m: {} for m in metrics}

    for f_name in filters_order:
        if f_name == 'Rohdaten':
            for m in metrics:
                p_values[m][f_name] = np.nan
            continue

        f_df = (
            res_df[res_df['Filter'] == f_name]
            .sort_values('Sensor')
            .reset_index(drop=True)
        )
        for m in metrics:
            _, p_val = ttest_rel(f_df[m], raw_df[m])
            p_values[m][f_name] = p_val

    # --- EXCEL 1: Einzelergebnisse pro Sensor ---
    res_df.to_excel('Filter_Vergleich_Einzelergebnisse.xlsx', index=False)

    # --- EXCEL 2: Zusammenfassung inkl. p-Values für die Masterarbeit ---
    export_rows = []
    for f_name in filters_order:
        row = {'Filter': f_name}
        for m in metrics:
            mean_val = avg_summary.loc[f_name, m]
            std_val = std_summary.loc[f_name, m]
            p_val = p_values[m][f_name]

            row[f'{m}_Mean'] = mean_val
            row[f'{m}_Std'] = std_val
            row[f'{m}_p_value'] = p_val
            row[f'{m}_Signifikant'] = get_p_asterisk(p_val)
        export_rows.append(row)

    excel_summary_df = pd.DataFrame(export_rows)
    excel_summary_df.to_excel('Filter_Zusammenfassung_Masterarbeit.xlsx', index=False)
    print('Tabellen wurden als Excel-Dateien gespeichert.')

    # --- DIAGRAMMERSTELLUNG ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    colors = ['#7f7f7f', '#1f77b4', '#2ca02c', '#d62728']

    plot_configs = [
        (axes[0, 0], 'SNR_dB', 'SNR [dB]', 'Signal-Rausch-Verhältnis (SNR)', True),
        (axes[0, 1], 'R2_Linearitaet', r'Bestimmtheitsmaß $R^2$ [-]', 'Linearität ($R^2$)', True),
        (axes[1, 0], 'Lin_Abweichung_pct_FSO', 'Max. Abweichung [% FSO]', 'Max. Linearitätsabweichung', False),
        (axes[1, 1], 'Signalamplitude_uH', 'Amplitude [µH]', 'Signalamplitude', 'any')
    ]

    for ax, metric, ylabel, title, higher_is_better in plot_configs:
        y_values = avg_summary[metric]
        y_errors = std_summary[metric]

        bars = ax.bar(avg_summary.index, y_values, yerr=y_errors, capsize=4, color=colors, alpha=0.85)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=20)

        max_err_top = (y_values + y_errors).max()
        min_err_bottom = (y_values - y_errors).min()

        if metric == 'R2_Linearitaet':
            ax.set_ylim(max(0, min_err_bottom - 0.02), 1.02)
        else:
            ax.set_ylim(0, max_err_top * 1.15)

        for bar, f_name in zip(bars, avg_summary.index):
            if f_name == 'Rohdaten':
                continue

            p_val = p_values[metric][f_name]
            raw_val = avg_summary.loc['Rohdaten', metric]
            curr_val = avg_summary.loc[f_name, metric]

            should_show_star = False

            if higher_is_better == 'any':
                if not np.isnan(p_val) and p_val < 0.05:
                    should_show_star = True
            else:
                is_improvement = False
                if higher_is_better is True and curr_val > raw_val:
                    is_improvement = True
                elif higher_is_better is False and curr_val < raw_val:
                    is_improvement = True

                if is_improvement and not np.isnan(p_val) and p_val < 0.05:
                    should_show_star = True

            if should_show_star:
                y_top = bar.get_height() + std_summary.loc[f_name, metric]
                y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
                offset = y_span * 0.015

                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_top + offset,
                    '*',
                    ha='center',
                    va='bottom',
                    fontsize=14,
                    fontweight='bold',
                    color='black'
                )

    plt.suptitle('Einfluss der Filter auf die Messwerte (Mittelwert ± Std, * p < 0.05 vs. Rohdaten)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_filter_vergleich_real.svg', dpi=300)
    plt.show()


if __name__ == '__main__':
    evaluate_filters_on_excel_uH('Zusammenfassung_1000_ohne_oval.xlsx')

