# Anamoly-detectection-using-iot-sensor-data
Deep learning models such as Autoencoders and Long Short-Term Memory (LSTM) networks are particularly suitable for time-series anomaly detection. Autoencoders learn to reconstruct normal patterns of data, and deviations in reconstruction errors signal anomalies. 
!pip install openvino-dev tensorflow scikit-learn pandas matplotlib -q
import pandas as pd

# Load the dataset
# Replace 'your_dataset.csv' with the actual path to your dataset file
try:
    df = pd.read_csv('your_dataset.csv')
    print("Dataset loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print("Error: Dataset file not found. Please upload the dataset or provide the correct file path.")
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
import numpy as np

def generate_iot_data(n_samples=5000, n_features=10):
    """Generate synthetic IoT sensor data"""
    np.random.seed(42)

    # Normal data with patterns
    time_steps = np.linspace(0, 50, n_samples)
    X = np.zeros((n_samples, n_features))

    for i in range(n_features):
        base = 20 + 5 * np.sin(2 * np.pi * time_steps / 24)
        noise = np.random.normal(0, 0.5, n_samples)
        X[:, i] = base + noise

    y = np.zeros(n_samples)
    anomaly_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)

    for idx in anomaly_indices:
        sensor_idx = np.random.randint(n_features)
        X[idx, sensor_idx] *= np.random.choice([0.1, 5])  # Spike anomaly
        y[idx] = 1

    return X, y
    from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Shape of training data (X_train):", X_train.shape)
print("Shape of testing data (X_test):", X_test.shape)
print("Shape of training labels (y_train):", y_train.shape)
print("Shape of testing labels (y_test):", y_test.shape)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Define the autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 5 #latent dimension

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

# Train the model
history = autoencoder.fit(X_train, X_train,
                          epochs=100,
                          batch_size=32,
                          shuffle=True,
                          validation_split=0.2,
                          callbacks=[early_stopping])

print("Model training complete.")
from sklearn.metrics import accuracy_score, f1_score

# Assuming 'autoencoder' model is already trained from previous steps
# Assuming X_test and y_test are available from previous steps

print(" Evaluating the autoencoder...")

# Calculate reconstructions and reconstruction errors on the test set
test_reconstructions = autoencoder.predict(X_test)
test_mse = np.mean(np.power(X_test - test_reconstructions, 2), axis=1)


# To get the threshold from the training data, we need to predict on X_train as well
train_reconstructions = autoencoder.predict(X_train)
train_mse = np.mean(np.power(X_train - train_reconstructions, 2), axis=1)
threshold = np.percentile(train_mse, 95)


# Classify as anomaly if reconstruction error is above the threshold
predictions = (test_mse > threshold).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"✅ Autoencoder Model Evaluation - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
import matplotlib.pyplot as plt

# Plot reconstruction errors for normal and anomaly data
plt.figure(figsize=(12, 6))
plt.hist(test_mse[y_test == 0], bins=50, color='blue', alpha=0.7, label='Normal')
plt.hist(test_mse[y_test == 1], bins=50, color='red', alpha=0.7, label='Anomaly')
plt.axvline(threshold, color='k', linestyle='dashed', linewidth=2, label='Threshold')
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Frequency")
plt.title("Distribution of Reconstruction Errors")
plt.legend()
plt.show()

