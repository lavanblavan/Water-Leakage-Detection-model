import numpy as np
import pandas as pd
import scipy.signal as signal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import re
import joblib

# Define file paths
file_paths = {
    "NL":r"no_leak.csv" ,
    "Leak": 
    [
        
        r"leak1.csv",
        r"leak2.csv",
    ]
}

# Function to compute PSD features using Welch's method
def compute_psd(data, fs=1000, nperseg=64):
    psd_features = []
    for i in range(data.shape[1]):  # Compute PSD for each axis (X, Y, Z)
        freqs, psd_values = signal.welch(data[:, i], fs=fs, nperseg=nperseg)
        psd_features.extend(psd_values)  # Append PSD features for each axis
    return np.array(psd_features)

# Load dataset and segment into parts
X, y = [], []
for label, paths in file_paths.items():
    if isinstance(paths, str):
        paths = [paths]  # Ensure it's a list

    for path in paths:
        with open(path, 'r') as f:  # Use with open to correctly handle files
            lines = f.readlines()

        signal_data = []

        for line in lines:
            try:
                match = re.findall(r"X:\s*(-?\d+\.\d+),\s*Y:\s*(-?\d+\.\d+),\s*Z:\s*(-?\d+\.\d+)", line)
                if match:
                    x, y_val, z = map(float, match[0])  # Convert to float
                    signal_data.append([x, y_val, z])
            except ValueError:
                print(f"Skipping line due to invalid format: {line.strip()}")  # Improved error message
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        signal_data = np.array(signal_data)  # Convert to numpy array after processing

        # Define segment size
        segment_size = max(1, len(signal_data) // 1000)  # Split into 20 segments

        for i in range(0, len(signal_data), segment_size):
            segment = signal_data[i:i + segment_size]
            if len(segment) < segment_size:  # Skip incomplete segments
                continue

            psd_features = compute_psd(segment)  # Compute PSD features
            X.append(psd_features)
            y.append(0 if label == "NL" else 1)  # 0 for No Leak, 1 for Leak

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)
print(f"Total samples: {len(X)}")

# Convert labels to categorical (binary classification)
y = to_categorical(y)

# Normalize PSD values
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')


# Reshape for CNN input (1D signal with 1 channel)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Define CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=X_train.shape[1:]),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Binary classification
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test),
                    class_weight=class_weights, callbacks=[early_stopping])

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {acc:.2f}")
model.save('my_model.h5')

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()