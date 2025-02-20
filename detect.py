import numpy as np
import scipy.signal as signal
import time
import tensorflow as tf
import joblib
import serial  # Import pyserial for communication

# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

# Load the saved scaler
scaler = joblib.load('scaler.pkl')

# Serial port configuration (adjust as needed)
SERIAL_PORT = "COM3"  # Change this to the correct port (e.g., "/dev/ttyUSB0" for Linux/Mac)
BAUD_RATE = 115200  # Ensure this matches your microcontroller's baud rate

# Open serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit()

# Function to compute PSD features
def compute_psd(data, fs=1000, nperseg=64):
    psd_features = []
    for i in range(data.shape[1]):  # Compute PSD for each axis (X, Y, Z)
        freqs, psd_values = signal.welch(data[:, i], fs=fs, nperseg=nperseg)
        psd_features.extend(psd_values)  # Append PSD features for each axis
    return np.array(psd_features)

# Function to preprocess real-time data
def preprocess_realtime_data(signal_data, scaler):
    psd_features = compute_psd(signal_data)
    psd_features = scaler.transform([psd_features])  # Normalize
    psd_features = psd_features.reshape(psd_features.shape[0], psd_features.shape[1], 1)  # Reshape for CNN
    return psd_features

# Read data from the microcontroller over serial
def read_serial_data():
    signal_data = []
    
    for _ in range(1000):  # Collect 1000 samples for a full window
        try:
            line = ser.readline().decode('utf-8').strip()
            
            # Expecting format: X:0.12,Y:-0.03,Z:0.98
            if line:
                values = line.split(',')
                if len(values) == 3:
                    x, y, z = [float(v.split(':')[1]) for v in values]
                    signal_data.append([x, y, z])
        except Exception as e:
            print(f"Error reading serial data: {e}")
    
    return np.array(signal_data)

# Main real-time leak detection loop
def realtime_leak_detection(model, scaler):
    while True:
        try:
            print("Reading data from microcontroller...")
            realtime_signal_data = read_serial_data()  # Read from serial
            
            if len(realtime_signal_data) == 0:
                print("No valid data received. Skipping iteration.")
                continue
            
            preprocessed_data = preprocess_realtime_data(realtime_signal_data, scaler)

            prediction = model.predict(preprocessed_data)
            predicted_class = np.argmax(prediction, axis=1)[0]

            if predicted_class == 0:
                print("No Leak Detected")
            else:
                print("Leak Detected!")

            time.sleep(1)  # Adjust based on sampling rate

        except KeyboardInterrupt:
            print("\nStopping real-time leak detection.")
            ser.close()  # Close serial connection before exiting
            break

# Start real-time detection
if __name__ == "__main__":
    print("Starting real-time leak detection...")
    realtime_leak_detection(model, scaler)
