
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Download stock data
stock_symbol = 'AAPL'
data = yf.download(stock_symbol, start='2020-01-01', end='2023-01-01')

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Define the autoencoder model
input_dim = (seq_length, 1)
inputs = Input(shape=input_dim)

# Encoder
encoded = LSTM(128, activation='relu', return_sequences=True)(inputs)
encoded = LSTM(64, activation='relu')(encoded)
encoded = RepeatVector(seq_length)(encoded)

# Decoder
decoded = LSTM(64, activation='relu', return_sequences=True)(encoded)
decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(1))(decoded)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = autoencoder.fit(
    X, X, epochs=100, batch_size=32, validation_split=0.2,
    callbacks=[early_stopping]
)

# Predict and calculate reconstruction error
predictions = autoencoder.predict(X)
reconstruction_error = np.mean(np.square(X - predictions), axis=1)

# Set a threshold for anomaly detection
threshold = np.percentile(reconstruction_error, 95)
anomalies = reconstruction_error > threshold

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(reconstruction_error, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()

# Highlight anomalies in the original data
anomaly_indices = np.where(anomalies)[0] + seq_length
anomaly_dates = data.index[anomaly_indices]

# Plot the stock price with anomalies highlighted
plt.figure(figsize=(14, 5))
plt.plot(data['Close'], label='Stock Price')
plt.scatter(anomaly_dates, data.loc[anomaly_dates, 'Close'], color='r', label='Anomalies')
plt.legend()
plt.show()

# Calculate validation metrics
mse = mean_squared_error(X.flatten(), predictions.flatten())
mae = mean_absolute_error(X.flatten(), predictions.flatten())
r2 = r2_score(X.flatten(), predictions.flatten())

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")