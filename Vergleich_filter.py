import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import stats


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
        window_sg = 51
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

        # Exakt die ersten 3,0 Sekunden Ruhemessung verwenden
        zero_mask = (t <= 3.0) & (strain < 0.05)
        if np.sum(zero_mask) < 10:
            zero_mask = t <= 3.0

        peaks, _ = signal.find_peaks(strain, distance=100, height=1.8)
        valleys, _ = signal.find_peaks(-strain, distance=100, height=-0.2)
        min_len = min(len(peaks), len(valleys))

        for f_name, f_ind in filters.items():
            # Standardabweichung des Rauschens in µH
            noise_std = f_ind[zero_mask].std()

            # Signalamplitude in µH
            amp_mean = np.mean(
                np.abs(f_ind[peaks[:min_len]] - f_ind[valleys[:min_len]])
            )

            # SNR [dB]
            snr_db = (
                20 * np.log10(amp_mean / noise_std)
                if noise_std > 0
                else np.nan
            )

            # Linearität R²
            p = np.polyfit(strain, f_ind, 1)
            pred = np.polyval(p, strain)
            ss_res = np.sum((f_ind - pred) ** 2)
            ss_tot = np.sum((f_ind - np.mean(f_ind)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

            # Max. Linearitätsabweichung in % FSO
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

    # =========================================================================
    # ERSTELLUNG UND NEUBERECHNUNG VON TABELLE 8 (INKL. P-WERTE JE VERGLEICH)
    # =========================================================================
    filters_order = [
        'Rohdaten',
        'Savitzky-Golay',
        'Butterworth',
        'gleitender Mittelwert',
    ]
    metrics = {
        'SNR_dB': 'SNR [dB]',
        'R2_Linearitaet': 'Bestimmtheitsmaß R²',
        'Lin_Abweichung_pct_FSO': 'Max. Abweichung [% FSO]',
        'Signalamplitude_uH': 'Signalamplitude [µH]',
    }

    raw_data = res_df[res_df['Filter'] == 'Rohdaten']

    tab8_rows = []

    for metric_key, metric_name in metrics.items():
        row = {'Kennwert': metric_name}

        # Referenzwerte (Rohdaten) für gepaarten t-Test
        raw_vals = raw_data[metric_key].values.astype(float)

        for f_name in filters_order:
            filt_vals = res_df[res_df['Filter'] == f_name][
                metric_key
            ].values.astype(float)

            mean_val = np.mean(filt_vals)
            std_val = np.std(filt_vals, ddof=1)

            # Formatierung je nach Kennwert
            if metric_key == 'R2_Linearitaet':
                val_str = f'{mean_val:.3f} ± {std_val:.3f}'
            elif metric_key == 'Signalamplitude_uH':
                val_str = f'{mean_val:.4f} ± {std_val:.4f}'
            else:
                val_str = f'{mean_val:.2f} ± {std_val:.2f}'

            # p-Wert des Einzelvergleichs zum Ausgangswert (Rohdaten)
            if f_name == 'Rohdaten':
                row[f_name] = val_str
            else:
                _, p_val = stats.ttest_rel(
                    raw_vals, filt_vals, nan_policy='omit'
                )
                p_str = f'p = {p_val:.3f}' if p_val >= 0.001 else 'p < 0.001'
                star = ' *' if p_val < 0.05 else ''
                row[f_name] = f'{val_str} ({p_str}){star}'

        # Haupt-p-Wert über alle Verfahren (Einfaktorielle ANOVA)
        all_filt_data = [
            res_df[res_df['Filter'] == f][metric_key].values.astype(float)
            for f in filters_order
        ]
        _, row['p-Wert (ANOVA)'] = stats.f_oneway(*all_filt_data)

        tab8_rows.append(row)

    tab8_df = pd.DataFrame(tab8_rows)

    print('\n=============================================================')
    print(' TABELLE 8: Einfluss der Filterverfahren auf die Kennwerte')
    print('=============================================================')
    print(tab8_df.to_string(index=False))
    print(
        '\n(* Kennzeichnet signifikante Unterschiede gegenüber den Rohdaten, p < 0.05)\n'
    )

    # Export von Tabelle 8 und den Einzelergebnissen als Excel
    tab8_df.to_excel('Tabelle_8_Filtervergleich.xlsx', index=False)
    res_df.to_excel('Filter_Vergleich_Auswertung.xlsx', index=False)

    # =========================================================================
    # DIAGRAMMERSTELLUNG
    # =========================================================================
    avg_summary = res_df.groupby('Filter')[list(metrics.keys())].mean()
    std_summary = res_df.groupby('Filter')[list(metrics.keys())].std()

    avg_plot = avg_summary.reindex(filters_order)
    std_plot = std_summary.reindex(filters_order)
    colors = ['#7f7f7f', '#3182bd', '#41ab5d', '#de2d26']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    def annotate_significance(ax, metric_name):
        raw_vals = res_df[res_df['Filter'] == 'Rohdaten'][
            metric_name
        ].values.astype(float)

        for idx, filter_name in enumerate(filters_order):
            if filter_name == 'Rohdaten':
                continue

            filt_vals = res_df[res_df['Filter'] == filter_name][
                metric_name
            ].values.astype(float)

            _, p_val = stats.ttest_rel(raw_vals, filt_vals, nan_policy='omit')

            if p_val < 0.05:
                mean_val = avg_plot.loc[filter_name, metric_name]
                std_val = std_plot.loc[filter_name, metric_name]

                y_offset = std_val * 0.15 if std_val > 0 else mean_val * 0.02
                y_pos = mean_val + std_val + y_offset

                ax.text(
                    idx,
                    y_pos,
                    '*',
                    ha='center',
                    va='bottom',
                    fontsize=14,
                    fontweight='bold',
                )

    # Plot 1: SNR
    axes[0, 0].bar(
        avg_plot.index,
        avg_plot['SNR_dB'],
        yerr=std_plot['SNR_dB'],
        capsize=4,
        color=colors,
    )
    axes[0, 0].set_ylabel('SNR [dB]', fontweight='bold')
    axes[0, 0].set_title('Signal-Rausch-Verhältnis (SNR)', fontweight='bold')
    axes[0, 0].set_ylim(0, 50)
    axes[0, 0].tick_params(axis='x', rotation=20)
    annotate_significance(axes[0, 0], 'SNR_dB')

    # Plot 2: Linearität (R²)
    axes[0, 1].bar(
        avg_plot.index,
        avg_plot['R2_Linearitaet'],
        yerr=std_plot['R2_Linearitaet'],
        capsize=4,
        color=colors,
    )
    axes[0, 1].set_ylabel(r'Bestimmtheitsmaß $R^2$ [-]', fontweight='bold')
    axes[0, 1].set_title(r'Linearität ($R^2$)', fontweight='bold')
    axes[0, 1].set_ylim(0.85, 1.02)
    axes[0, 1].tick_params(axis='x', rotation=20)
    annotate_significance(axes[0, 1], 'R2_Linearitaet')

    # Plot 3: Max. Linearitätsabweichung
    axes[1, 0].bar(
        avg_plot.index,
        avg_plot['Lin_Abweichung_pct_FSO'],
        yerr=std_plot['Lin_Abweichung_pct_FSO'],
        capsize=4,
        color=colors,
    )
    axes[1, 0].set_ylabel('Max. Abweichung [% FSO]', fontweight='bold')
    axes[1, 0].set_title('Max. Linearitätsabweichung', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=20)
    annotate_significance(axes[1, 0], 'Lin_Abweichung_pct_FSO')

    # Plot 4: Signalamplitude
    axes[1, 1].bar(
        avg_plot.index,
        avg_plot['Signalamplitude_uH'],
        yerr=std_plot['Signalamplitude_uH'],
        capsize=4,
        color=colors,
    )
    axes[1, 1].set_ylabel('Amplitude [µH]', fontweight='bold')
    axes[1, 1].set_title('Signalamplitude', fontweight='bold')
    axes[1, 1].set_ylim(0, 0.008)
    axes[1, 1].tick_params(axis='x', rotation=20)
    annotate_significance(axes[1, 1], 'Signalamplitude_uH')

    plt.tight_layout()
    plt.savefig('fig_filter_vergleich_real.svg', dpi=300)
    plt.show()


if __name__ == '__main__':
    evaluate_filters_on_excel_uH('Zusammenfassung_1000_ohne_oval.xlsx')