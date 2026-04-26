"""
==============================================================================
STEP 4: PRODUCTION MONITORING
==============================================================================
- Purpose: This is the script you leave running day-to-day once the machine 
  is operating in the factory or real environment.
  
- How it works: It doesn't use labels because in the real world we don't 
  know upfront if the machine is healthy or failing. Instead of a fixed file, 
  it generates a new CSV every day with the date (e.g. 25_10_2023.csv) to keep 
  a historical record organized and prevent the Raspberry Pi's memory from 
  collapsing. It also sends data in real-time to an InfluxDB database for 
  visualization in Grafana.
==============================================================================
"""
import time
import csv
import numpy as np
from scipy.stats import kurtosis
from smbus2 import SMBus
from influxdb import InfluxDBClient
from datetime import datetime

# ==========================================
# CONFIGURATION (InfluxDB v1.6.7)
# ==========================================
INFLUX_HOST = 'localhost'
INFLUX_PORT = 8086
INFLUX_DB = 'vibrations'

# Generate the filename dynamically with today's date
today_date = datetime.now().strftime("%d_%m_%Y")
CSV_FILE = f"{today_date}.csv"

# ==========================================
# PHYSICAL PARAMETERS 
# ==========================================
FS = 100               # 100 Hz sampling rate
SAMPLING_TIME = 1.0 / FS
WINDOW_SIZE = 200      # 2 seconds of data per window
OVERLAP = 100          # 50% overlap

MPU_ADDRESS = 0x68
PWR_MGMT_1_REG = 0x6B
ACCEL_ZOUT_H_REG = 0x3F 

# ==========================================
# FUNCTIONS
# ==========================================
def initialize_sensor(bus):
    """
    Initializes the MPU6050 sensor by writing to its power management register
    to wake it up from sleep mode.
    """
    bus.write_byte_data(MPU_ADDRESS, PWR_MGMT_1_REG, 0)

def read_z_acceleration(bus):
    """
    Reads raw Z-axis data from the MPU6050 accelerometer and converts it 
    to 'g' (G-force) values.
    """
    high = bus.read_byte_data(MPU_ADDRESS, ACCEL_ZOUT_H_REG)
    low = bus.read_byte_data(MPU_ADDRESS, ACCEL_ZOUT_H_REG + 1)
    value = (high << 8) | low
    if value > 32768:
        value = value - 65536
    return value / 16384.0

def extract_mechanical_features(signal):
    """
    Processes a time window of vibration data and extracts key features
    in the time and frequency domains using FFT.
    
    Returns a list with: [RMS, Crest Factor, Kurtosis, 0-10Hz Energy, 10-30Hz Energy, 30-50Hz Energy]
    """
    rms = np.sqrt(np.mean(signal**2))
    max_peak = np.max(np.abs(signal))
    crest_factor = max_peak / rms if rms > 0 else 0
    kurtosis_val = kurtosis(signal)
    
    fft_values = np.fft.fft(signal)
    amplitudes = np.abs(fft_values[0:WINDOW_SIZE//2])
    frequencies = np.fft.fftfreq(WINDOW_SIZE, 1/FS)[0:WINDOW_SIZE//2]
    
    low_energy = np.sum(amplitudes[(frequencies > 0) & (frequencies <= 10)]**2)
    mid_energy = np.sum(amplitudes[(frequencies > 10) & (frequencies <= 30)]**2)
    high_energy = np.sum(amplitudes[(frequencies > 30) & (frequencies <= 50)]**2)
    
    return [rms, crest_factor, kurtosis_val, low_energy, mid_energy, high_energy]

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    bus = SMBus(1)
    initialize_sensor(bus)
    influx_client = InfluxDBClient(host=INFLUX_HOST, port=INFLUX_PORT, database=INFLUX_DB)
    
    # Open or create the file for the current day
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Only write the header if the file is empty
        if f.tell() == 0:
            writer.writerow(['RMS', 'Crest', 'Kurtosis', 'Low_E_0_10Hz', 'Mid_E_10_30Hz', 'High_E_30_50Hz'])

    z_window = []
    print(f"🚀 Starting monitoring. Saving to: {CSV_FILE}")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            loop_start = time.time()
            z_g = read_z_acceleration(bus)
            z_window.append(z_g)
            
            json_body_raw = [{"measurement": "motor_vibration", "fields": {"z_axis": z_g}}]
            influx_client.write_points(json_body_raw)
            
            if len(z_window) == WINDOW_SIZE:
                signal_np = np.array(z_window)
                features = extract_mechanical_features(signal_np)
                
                # 1. Save to daily CSV
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(features)
                
                # 2. Send features to Grafana
                json_features = [{
                    "measurement": "mechanical_features",
                    "fields": {
                        "RMS": float(features[0]),
                        "Crest": float(features[1]),
                        "Kurtosis": float(features[2]),
                        "Low_E": float(features[3]),
                        "Mid_E": float(features[4]),
                        "High_E": float(features[5])
                    }
                }]
                influx_client.write_points(json_features)
                
                print(f"Processed window -> RMS: {features[0]:.4f}g | Kurtosis: {features[2]:.2f}")
                z_window = z_window[OVERLAP:]

            elapsed_time = time.time() - loop_start
            wait_time = SAMPLING_TIME - elapsed_time
            if wait_time > 0:
                time.sleep(wait_time)

    except KeyboardInterrupt:
        print(f"\nMonitoring stopped. Data saved in {CSV_FILE}.")
        influx_client.close()
        bus.close()