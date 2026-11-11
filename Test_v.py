
import time
import math
import csv
import threading
import serial
from pipython import GCSDevice

# ===================================================================
# CONFIGURATION
# ===================================================================
PI_COM_PORT = 4
PI_BAUDRATE = 115200
PI_AXIS = 1

ARDUINO_PORT = "COM8"
ARDUINO_BAUDRATE = 115200

CAPACITANCE_F = 1.1e-9      # 1.1 nF
SPEED_MM_S = 2.0            # 2.0 mm/s
MOVE_DIST_MM = 2.0         # 2.0 mm
CYCLES = 1000
PRE_RUN_TIME_S = 3.0
POST_RUN_TIME_S = 3.0

SAMPLE_RATE_HZ = 80.0
DT = 1.0 / SAMPLE_RATE_HZ   # 0.0125 s


# ===================================================================
# THREAD-SICHERER ESP32 EMPFÄNGER
# ===================================================================
class ESP32Receiver(threading.Thread):
    def __init__(self, port, baudrate):
        super().__init__()
        self.ser = serial.Serial(port, baudrate, timeout=1.0)
        self.running = False
        self.latest_freq = None
        self.lock = threading.Lock()
        time.sleep(2.0)
        self.ser.reset_input_buffer()

    def run(self):
        self.running = True
        while self.running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line.startswith("FREQ:"):
                    val = float(line.split(":")[1])
                    if val > 0:
                        with self.lock:
                            self.latest_freq = val
            except Exception:
                pass

    def stop(self):
        self.running = False
        if self.ser.is_open:
            self.ser.close()

    def get_freq(self):
        with self.lock:
            return self.latest_freq


# ===================================================================
# HAUPTPROGRAMM
# ===================================================================
def main():
    receiver = ESP32Receiver(ARDUINO_PORT, ARDUINO_BAUDRATE)
    receiver.start()

    try:
        print("Warte auf LDC1101 Sensordaten vom ESP32-S3...")
        while receiver.get_freq() is None:
            time.sleep(0.05)

        print(f"ESP32 sendet! Erste Frequenz: {receiver.get_freq():.2f} Hz")
        print("Starte PI-Steuerung...")

        log_records = []

        with GCSDevice('C-863') as pidevice:
            pidevice.ConnectRS232(PI_COM_PORT, PI_BAUDRATE)
            pidevice.SVO(PI_AXIS, True)
            pidevice.VEL(PI_AXIS, SPEED_MM_S)

            # Startposition EINMALIG abfragen
            start_pos = pidevice.qPOS(PI_AXIS)[PI_AXIS]
            target_low = start_pos - MOVE_DIST_MM
            move_duration = MOVE_DIST_MM / SPEED_MM_S

            t_global_start = time.perf_counter()

            def record_sample():
                # Zeitstempel unmittelbar vor den Messungen
                t_now = time.perf_counter() - t_global_start

                # Aktuelle Sensorfrequenz vom ESP32
                freq = receiver.get_freq()

                # Tatsächliche Position vom PI-Controller abfragen
                current_pos = pidevice.qPOS(PI_AXIS)[PI_AXIS]

                # Induktivität berechnen
                inductance = None
                if freq and freq > 0:
                    inductance = 1.0 / (
                        (2.0 * math.pi * freq) ** 2 * CAPACITANCE_F
                    )

                # Messwerte gemeinsam speichern
                log_records.append(
                    (t_now, current_pos, freq, inductance)
                )

            # -------------------------------------------------------
            # PHASE 1: VORLAUF (3 Sek. Stillstand)
            # -------------------------------------------------------
            t_phase = time.perf_counter()

            while time.perf_counter() - t_phase < PRE_RUN_TIME_S:
                record_sample()
                time.sleep(DT)

            # -------------------------------------------------------
            # PHASE 2: ZYKLEN
            # -------------------------------------------------------
            for cycle in range(CYCLES):
                print(f"Zyklus {cycle + 1}/{CYCLES} läuft...")

                # 1. Rückfahrt (start_pos -> target_low)
                pidevice.MOV(PI_AXIS, target_low)
                t_phase = time.perf_counter()

                while True:
                    elapsed = time.perf_counter() - t_phase

                    if elapsed >= move_duration:
                        break

                    record_sample()
                    time.sleep(DT)

                # 2. Vorfahrt (target_low -> start_pos)
                pidevice.MOV(PI_AXIS, start_pos)
                t_phase = time.perf_counter()

                while True:
                    elapsed = time.perf_counter() - t_phase

                    if elapsed >= move_duration:
                        break

                    record_sample()
                    time.sleep(DT)

            # -------------------------------------------------------
            # PHASE 3: NACHLAUF (3 Sek. Stillstand)
            # -------------------------------------------------------
            t_phase = time.perf_counter()

            while time.perf_counter() - t_phase < POST_RUN_TIME_S:
                record_sample()
                time.sleep(DT)

    finally:
        # Empfänger stoppen
        receiver.stop()
        receiver.join()

    # =================================================================
    # CSV AUSWERTUNG & EXPORT
    # =================================================================
    print("Berechne I0 und speichere CSV...")

    pre_run_L = [
        rec[3] * 1e6
        for rec in log_records
        if rec[0] < PRE_RUN_TIME_S and rec[3] is not None
    ]

    I_0 = sum(pre_run_L) / len(pre_run_L) if pre_run_L else 0.0

    with open("messung_qPOS_jeder_Messpunkt.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")

        writer.writerow([
            "time_s",
            "position_mm",
            "strain_mm",
            "frequency_real_Hz",
            "inductance_uH",
            "delta_I_uH"
        ])

        for t_val, pos_val, freq_val, L_val in log_records:
            strain_mm = start_pos - pos_val

            freq_real_str = (
                f"{freq_val:.3f}".replace('.', ',')
                if freq_val else ""
            )

            if L_val is not None:
                L_uH_val = L_val * 1e6
                delta_I_val = L_uH_val - I_0

                L_uH_str = f"{L_uH_val:.9f}".replace('.', ',')
                delta_I_str = f"{delta_I_val:.9f}".replace('.', ',')
            else:
                L_uH_str = ""
                delta_I_str = ""

            writer.writerow([
                f"{t_val:.4f}".replace('.', ','),
                f"{pos_val:.3f}".replace('.', ','),
                f"{strain_mm:.3f}".replace('.', ','),
                freq_real_str,
                L_uH_str,
                delta_I_str
            ])

    print("Messung beendet! CSV erfolgreich gespeichert.")


if __name__ == "__main__":
    main()
