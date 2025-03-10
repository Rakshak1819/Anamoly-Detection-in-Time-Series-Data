# Anamoly-Detection-in-Time-Series-Data

This project demonstrates an anomaly detection framework for time series data using an LSTM autoencoder. The approach is applied to stock market data (AAPL) obtained via the `yfinance` library, where the model identifies anomalous patterns based on reconstruction error.

## Overview

The project downloads historical stock data for Apple Inc. from 2020 to 2023, preprocesses the closing price data using MinMax scaling, and creates sequences from the time series. An LSTM autoencoder is then built and trained to reconstruct these sequences. The reconstruction error is computed for each sequence and used to flag anomalies based on a defined threshold (the 95th percentile of errors). The results are visualized using line plots, with anomalies highlighted on the stock price chart.

## Prerequisites

- Python 3.7 or later
- The following Python libraries (listed in `requirements.txt`):
  - `yfinance`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `tensorflow`
  - `matplotlib`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/Anamoly-Detection-in-Time-Series-Data.git
   cd Anamoly-Detection-in-Time-Series-Data


### Install dependencies:

Install the required packages using pip:

pip install -r requirements.txt

### Running the Project

To run the anomaly detection script, execute the Python file in your terminal or command prompt:

python anamoly_time_series.py


Note: Replace your_script_name.py with the actual name of your Python file if it is different.

## Project Structure
anamoly_time_series.py: Main Python script containing the code for downloading data, building the autoencoder, training, and anomaly detection.
requirements.txt: Lists all required Python packages.
Additional files (if any) can be included in the repository as the project evolves.
How It Works

#### Data Acquisition:
The script uses yfinance to download Apple Inc. stock data.

#### Data Preprocessing:
The 'Close' price of the stock is scaled using MinMax scaling, and sequences of a fixed length (60 time steps) are created from the scaled data.

#### Model Architecture:
An LSTM autoencoder is defined with two main parts:

Encoder: Encodes input sequences into a compressed representation.
Decoder: Attempts to reconstruct the original sequences from the encoded representation.

#### Training:
The autoencoder is trained with early stopping based on validation loss to avoid overfitting.

#### Anomaly Detection:
After training, the reconstruction error for each sequence is computed. Anomalies are flagged if the error exceeds the 95th percentile threshold.

#### Visualization:
The reconstruction errors and detected anomalies are visualized on plots to help understand the model's performance.

## Results

Upon running the project, you will see:

- A plot of reconstruction errors with the anomaly threshold indicated.
- A line chart of the stock closing prices with anomalies highlighted.
- Printed evaluation metrics including Mean Squared Error, Mean Absolute Error, and the R² score.

## Contributing

If you have ideas for improvements or fixes:

- Fork the repository.
- Create a new branch.
- Implement your changes.
- Submit a pull request.

