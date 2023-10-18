import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests

api_key = 'YOUR_API_KEY'

# Symbol for the stock you want to fetch (e.g., AAPL for Apple Inc.)
symbol = 'AAPL'

# API endpoint for time series data
endpoint = f'https://www.alphavantage.co/query'
params = {
    'function': 'TIME_SERIES_DAILY',
    'symbol': symbol,
    'apikey': api_key,
    'outputsize': 'full'  # 'full' for complete historical data
}

try:
    # Make a GET request to the Alpha Vantage API
    response = requests.get(endpoint, params=params)
    data = response.json()

    # Extract daily stock price data into a DataFrame
    time_series = data['Time Series (Daily)']
    df = pd.DataFrame(time_series).T
    df.index = pd.to_datetime(df.index)
    df['4. close'] = df['4. close'].astype(float)

    # Normalize the data
    scaler = MinMaxScaler()
    df['Close'] = scaler.fit_transform(df['4. close'].values.reshape(-1, 1))

    # Create sequences for training data
    def create_sequences(data, sequence_length):
        sequences = []
        targets = []
        for i in range(len(data) - sequence_length):
            sequence = data[i:i + sequence_length]
            target = data[i + sequence_length]
            sequences.append(sequence)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    sequence_length = 10  # Number of previous days to use for prediction
    X, y = create_sequences(df['Close'].values, sequence_length)

    # Split data into training and testing sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Inverse transform the predictions to get real stock prices
    predictions = scaler.inverse_transform(predictions)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[train_size + sequence_length:], df['4. close'][train_size + sequence_length:], label='True Stock Price', color='blue')
    plt.plot(df.index[train_size + sequence_length:], predictions, label='Predicted Stock Price', color='red')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

except Exception as e:
    print(f"An error occurred: {str(e)}")
