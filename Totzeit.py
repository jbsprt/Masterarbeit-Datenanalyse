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
SPEED_MM_S = 5.0            # 5.0 mm/s
MOVE_DIST_MM = 2.0          # 2.0 mm Weg
HOLD_TIME_S = 1.0           # 1.0 Sekunden Haltezeit an den Endpunkten
CYCLES = 10                 # 10 Zyklen

PRE_RUN_TIME_S = 3.0
POST_RUN_TIME_S = 3.0

SAMPLE_RATE_HZ = 200.0
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

        def record_sample(current_pos):
            t_now = time.perf_counter() - t_global_start
            freq = receiver.get_freq()

            inductance = None
            if freq and freq > 0:
                inductance = 1.0 / ((2.0 * math.pi * freq) ** 2 * CAPACITANCE_F)

            log_records.append((t_now, current_pos, freq, inductance))

        # --- Phase 1: Vorlauf (3 Sek. Stillstand) ---
        t_phase = time.perf_counter()
        while time.perf_counter() - t_phase < PRE_RUN_TIME_S:
            record_sample(start_pos)
            time.sleep(DT)

        # --- Phase 2: Zyklen ---
        for cycle in range(CYCLES):
            print(f"Zyklus {cycle + 1}/{CYCLES} läuft...")

            # 1. Ausfahren / Dehnen (start_pos -> target_low mit 5 mm/s)
            pidevice.MOV(PI_AXIS, target_low)
            t_phase = time.perf_counter()
            while True:
                elapsed = time.perf_counter() - t_phase
                if elapsed >= move_duration:
                    break
                current_pos = max(target_low, start_pos - (SPEED_MM_S * elapsed))
                record_sample(current_pos)
                time.sleep(DT)

            # 2. Haltephase oben (3.0 Sekunden bei gedrehter/gedehnter Position)
            t_phase = time.perf_counter()
            while time.perf_counter() - t_phase < HOLD_TIME_S:
                record_sample(target_low)
                time.sleep(DT)

            # 3. Einfahren / Rückfahrt (target_low -> start_pos mit 5 mm/s)
            pidevice.MOV(PI_AXIS, start_pos)
            t_phase = time.perf_counter()
            while True:
                elapsed = time.perf_counter() - t_phase
                if elapsed >= move_duration:
                    break
                current_pos = min(start_pos, target_low + (SPEED_MM_S * elapsed))
                record_sample(current_pos)
                time.sleep(DT)

            # 4. Haltephase unten (3.0 Sekunden bei Ausgangsposition)
            t_phase = time.perf_counter()
            while time.perf_counter() - t_phase < HOLD_TIME_S:
                record_sample(start_pos)
                time.sleep(DT)

        # --- Phase 3: Nachlauf (3 Sek. Stillstand) ---
        t_phase = time.perf_counter()
        while time.perf_counter() - t_phase < POST_RUN_TIME_S:
            record_sample(start_pos)
            time.sleep(DT)

    # Empfänger stoppen
    receiver.stop()
    receiver.join()

    # --- CSV Auswertung & Export ---
    print("Berechne I0 und speichere CSV...")
    pre_run_L = [rec[3] * 1e6 for rec in log_records if rec[0] < PRE_RUN_TIME_S and rec[3] is not None]
    I_0 = sum(pre_run_L) / len(pre_run_L) if pre_run_L else 0.0

    output_filename = "Totzeit_()33_30w_2mm_5mms.csv"

    with open(output_filename, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["time_s", "position_mm", "strain_mm", "frequency_real_Hz", "frequency_sensor_Hz", "inductance_uH", "delta_I_uH"])

        for t_val, pos_val, freq_val, L_val in log_records:
            strain_mm = start_pos - pos_val

            freq_real_str = f"{freq_val:.3f}".replace('.', ',') if freq_val else ""
            freq_sensor_str = f"{freq_val / 4.0:.3f}".replace('.', ',') if freq_val else ""

            if L_val:
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
                freq_sensor_str,
                L_uH_str,
                delta_I_str
            ])

    print(f"Messung beendet! CSV erfolgreich unter '{output_filename}' gespeichert.")

if __name__ == "__main__":
    main()