
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import tensorflow as tf
import tensorflow_probability as tfp
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors

print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Probability version: {tfp.__version__}")

# ====================== PHASE 1: DATA COLLECTION & PREPROCESSING ======================

def download_stock_data(symbol, start_date, end_date):
    """Download historical stock data from Yahoo Finance"""
    print(f"Downloading data for {symbol} from {start_date} to {end_date}...")

    # Download data
    temp_df = yf.download(symbol, start=start_date, end=end_date)

    # Create a new DataFrame with proper indexing
    df = pd.DataFrame()
    df['Date'] = temp_df.index
    df['Open'] = temp_df['Open'].values
    df['High'] = temp_df['High'].values
    df['Low'] = temp_df['Low'].values
    df['Close'] = temp_df['Close'].values
    df['Volume'] = temp_df['Volume'].values
    df['Adj Close'] = temp_df['Adj Close'].values if 'Adj Close' in temp_df.columns else temp_df['Close'].values

    print(f"Downloaded {len(df)} records.")
    return df

def fetch_news_sentiment(symbol, start_date, end_date):
    """
    Simulate fetching news sentiment for a stock
    """
    # For demonstration - generate random sentiment scores
    date_range = pd.date_range(start=start_date, end=end_date)
    sentiment_data = []

    # Initialize sentiment analyzer
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()

    # Simulate news headlines
    headlines = [
        f"{symbol} reports strong quarterly earnings",
        f"{symbol} misses analyst expectations",
        f"Analysts upgrade {symbol} stock rating",
        f"Economic uncertainty affects {symbol}",
        f"{symbol} announces new product line",
        f"Market downturn impacts {symbol}",
        f"{symbol} faces regulatory challenges",
        f"Positive outlook for {symbol} in coming quarters"
    ]

    for date in date_range:
        if date.weekday() < 5:  # Only business days
            # Randomly select headlines for this day
            daily_headlines = np.random.choice(headlines, size=np.random.randint(1, 3))

            # Calculate sentiment for each headline and average
            sentiment_scores = [sia.polarity_scores(headline)['compound'] for headline in daily_headlines]
            avg_sentiment = np.mean(sentiment_scores)

            sentiment_data.append({
                'Date': date,
                'Sentiment': avg_sentiment
            })

    sentiment_df = pd.DataFrame(sentiment_data)
    # Don't set index, keep as regular columns for easier merging
    return sentiment_df

def calculate_technical_indicators(df):
    """Calculate technical indicators mentioned in the synopsis"""
    # Simple Moving Averages (SMA)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # Exponential Moving Averages (EMA)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

    # Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=30).std()

    # Moving Average Crossover
    df['MA_Crossover'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)

    # Price to Moving Average Ratio
    df['Price_to_MA_Ratio'] = df['Close'] / df['SMA_50']

    return df

def preprocess_data(df, sentiment_df=None):
    """Preprocess the data for model training"""
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Merge with sentiment data if available
    if sentiment_df is not None:
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        df = pd.merge(df, sentiment_df, on='Date', how='left')
        df['Sentiment'] = df['Sentiment'].fillna(0)  # Fill missing sentiment

    # Calculate technical indicators
    df = calculate_technical_indicators(df)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Store in SQLite database
    conn = sqlite3.connect("stock_prediction.db")
    df.to_sql("stock_data", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Preprocessed data shape: {df.shape}")
    return df

def create_features_target(df, target_col='Close', n_steps=250):
    """Create features and target for model training with sequence data for LSTM"""
    # Select features
    features = ['Open', 'High', 'Low', 'Volume',
                'SMA_5', 'SMA_20', 'SMA_50', 'SMA_200',
                'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
                'RSI', 'BB_Upper', 'BB_Lower', 'Volatility']

    # Add sentiment if available
    if 'Sentiment' in df.columns:
        features.append('Sentiment')

    # Scale the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = scaler_X.fit_transform(df[features])
    y = scaler_y.fit_transform(df[[target_col]])

    # Create sequences for LSTM
    X_seq, y_seq = [], []
    for i in range(len(X) - n_steps):
        X_seq.append(X[i:i+n_steps])
        y_seq.append(y[i+n_steps])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, features

# ====================== PHASE 2: MODEL DEVELOPMENT ======================

def build_bayesian_lstm_model(input_shape):
    """Build a Bayesian LSTM model using TensorFlow Probability"""
    # Make sure TFP layers are compatible with Keras
    # We'll use a simpler approach to incorporate uncertainty

    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.InputLayer(input_shape=input_shape),

        # First LSTM layer
        tf.keras.layers.LSTM(units=64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),

        # Second LSTM layer
        tf.keras.layers.LSTM(units=32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),

        # Output layer with mean and log variance
        tf.keras.layers.Dense(2)  # 2 outputs: mean and log_variance
    ])

    return model

def negative_log_likelihood(y_true, y_pred):
    """Custom loss function for probabilistic model"""
    # Extract mean and log variance from the model output
    mean = y_pred[:, 0:1]
    log_var = y_pred[:, 1:2]

    # Calculate negative log likelihood
    return 0.5 * tf.reduce_mean(
        tf.exp(-log_var) * tf.square(y_true - mean) + log_var
    )

def build_lstm_baseline(input_shape):
    """Build a standard LSTM model as baseline"""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def fit_arima_model(series):
    """Fit ARIMA model"""
    try:
        model = ARIMA(series, order=(5, 1, 0))
        model_fit = model.fit()
        return model_fit
    except:
        print("Error fitting ARIMA model. Using fallback parameters.")
        model = ARIMA(series, order=(1, 0, 0))
        model_fit = model.fit()
        return model_fit

def fit_random_forest(X_train_2d, y_train):
    """Fit Random Forest model"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_2d, y_train)
    return model

# ====================== PHASE 3: TRAINING & EVALUATION ======================

def evaluate_models(X_train, X_test, y_train, y_test, scaler_y, original_df):
    """Train and evaluate all models"""
    results = {}

    # Reshape data for non-sequential models
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    # 1. Bayesian Neural Network (LSTM with uncertainty)
    print("\nTraining Bayesian Neural Network...")
    bnn_model = build_bayesian_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    bnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=negative_log_likelihood
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001
    )

    history_bnn = bnn_model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    print("\nTraining LSTM baseline...")
    lstm_model = build_lstm_baseline(input_shape=(X_train.shape[1], X_train.shape[2]))

    early_stopping_lstm = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    history_lstm = lstm_model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping_lstm],
        verbose=1
    )

    # Save the trained models
    bnn_model.save("bnn_model.h5")
    lstm_model.save("lstm_model.h5")

    return results, bnn_model, lstm_model


# ====================== PHASE 4: VISUALIZATION ======================

def visualize_results(results, y_test_inv, dates_test=None):
    """Visualize predictions and uncertainty estimates"""
    plt.figure(figsize=(14, 8))

    # Plot actual values
    plt.plot(y_test_inv, 'b-', label='Actual', linewidth=2)

    # Plot predictions for each model
    colors = {'BNN': 'r', 'LSTM': 'g', 'RF': 'm', 'ARIMA': 'c'}

    for model_name, metrics in results.items():
        plt.plot(metrics['predictions'], f"{colors[model_name]}-", label=f"{model_name} Prediction", alpha=0.7)

        # Add uncertainty bands for BNN
        if model_name == 'BNN':
            plt.fill_between(
                np.arange(len(metrics['predictions'])),
                metrics['predictions'].flatten() - 2 * metrics['uncertainty'].flatten(),
                metrics['predictions'].flatten() + 2 * metrics['uncertainty'].flatten(),
                color='r', alpha=0.2, label='95% Confidence Interval'
            )

    plt.title('Stock Price Prediction Comparison', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('stock_prediction_comparison.png', dpi=300)
    plt.show()

    # Plot training history for neural networks
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(results['BNN']['history'].history['loss'], label='BNN Training Loss')
    plt.plot(results['BNN']['history'].history['val_loss'], label='BNN Validation Loss')
    plt.title('Bayesian Neural Network Training', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(results['LSTM']['history'].history['loss'], label='LSTM Training Loss')
    plt.plot(results['LSTM']['history'].history['val_loss'], label='LSTM Validation Loss')
    plt.title('LSTM Training', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()

    # Plot uncertainty calibration
    if 'BNN' in results:
        errors = np.abs(results['BNN']['predictions'].flatten() - y_test_inv.flatten())
        uncertainty = results['BNN']['uncertainty'].flatten()

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(uncertainty, errors, alpha=0.5)
        plt.title('Uncertainty vs Prediction Error', fontsize=14)
        plt.xlabel('Predicted Uncertainty (Std Dev)', fontsize=12)
        plt.ylabel('Absolute Error', fontsize=12)

        # Add trend line
        z = np.polyfit(uncertainty, errors, 1)
        p = np.poly1d(z)
        plt.plot(uncertainty, p(uncertainty), "r--", alpha=0.8)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        # Create calibration curve
        sorted_indices = np.argsort(uncertainty)
        n_bins = 10
        bin_size = len(sorted_indices) // n_bins

        bin_uncertainties = []
        bin_errors = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_indices)
            bin_indices = sorted_indices[start_idx:end_idx]

            bin_uncertainties.append(np.mean(uncertainty[bin_indices]))
            bin_errors.append(np.mean(errors[bin_indices]))

        plt.plot(bin_uncertainties, bin_errors, 'bo-')
        plt.plot([0, max(bin_uncertainties)], [0, max(bin_uncertainties)], 'r--')
        plt.title('Uncertainty Calibration Plot', fontsize=14)
        plt.xlabel('Mean Predicted Uncertainty', fontsize=12)
        plt.ylabel('Mean Absolute Error', fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('uncertainty_analysis.png', dpi=300)
        plt.show()

    # Calculate prediction error distribution
    plt.figure(figsize=(12, 6))

    for model_name, metrics in results.items():
        errors = metrics['predictions'].flatten() - y_test_inv.flatten()

        plt.hist(errors, bins=30, alpha=0.5, label=f"{model_name}")

    plt.axvline(0, color='k', linestyle='--')
    plt.title('Prediction Error Distribution by Model', fontsize=14)
    plt.xlabel('Prediction Error ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300)
    plt.show()

# ====================== PHASE 5: FORECASTING FUTURE PRICES ======================

def forecast_future(bnn_model, last_sequence, scaler_X, scaler_y, days=30):
    """Forecast future stock prices using the BNN model"""
    # Initialize with the last observed sequence
    current_seq = last_sequence.copy()

    # Store predictions
    future_preds = []
    future_uncertainty = []

    # Make predictions for specified number of days
    for _ in range(days):
        # Get prediction with mean and log variance
        pred = bnn_model.predict(current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1]))
        pred_mean = pred[0, 0]
        pred_logvar = pred[0, 1]
        pred_std = np.sqrt(np.exp(pred_logvar))

        # Store prediction and uncertainty
        future_preds.append(pred_mean)
        future_uncertainty.append(pred_std)

        # Update sequence for next prediction (simple approach - shift and update close price)
        # In a full implementation, you'd generate all features
        new_row = np.zeros(current_seq.shape[1])
        new_row[0] = pred_mean  # Assuming first feature is close price

        # Shift sequence and add new prediction
        current_seq = np.vstack([current_seq[1:], new_row])

    # Convert predictions to original scale
    future_preds = np.array(future_preds).reshape(-1, 1)
    future_preds_inv = scaler_y.inverse_transform(future_preds)

    # Scale uncertainty
    future_uncertainty = np.array(future_uncertainty) * (scaler_y.data_max_ - scaler_y.data_min_)

    return future_preds_inv, future_uncertainty

def plot_forecast(last_actual_prices, forecasted_prices, uncertainty, symbol):
    """Plot the forecasted prices with uncertainty"""
    # Create date range for forecasting
    last_date = datetime.datetime.now()
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(len(forecasted_prices))]

    # Create date range for historical data
    historical_dates = [last_date - datetime.timedelta(days=len(last_actual_prices)-i) for i in range(len(last_actual_prices))]

    plt.figure(figsize=(14, 8))

    # Plot historical data
    plt.plot(historical_dates, last_actual_prices, 'b-', label='Historical Prices')

    # Plot forecasted data
    plt.plot(future_dates, forecasted_prices, 'r-', label='Forecasted Prices')

    # Add uncertainty bands
    plt.fill_between(
        future_dates,
        forecasted_prices.flatten() - 2 * uncertainty,
        forecasted_prices.flatten() + 2 * uncertainty,
        color='r', alpha=0.2, label='95% Confidence Interval'
    )

    plt.title(f'{symbol} Stock Price Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stock_price_forecast.png', dpi=300)
    plt.show()

# ====================== MAIN EXECUTION ======================
def predict_today_with_models(models_dict, processed_df, scaler_X, scaler_y, n_steps=60, features=None):
    """
    Get predictions for today from multiple models based on the most recent available data
    """
    print("\n===== TODAY'S PREDICTIONS FROM MULTIPLE MODELS =====")

    # Get the most recent n_steps days of data
    if features is None:
        features = ['Open', 'High', 'Low', 'Volume',
                    'SMA_5', 'SMA_20', 'SMA_50', 'SMA_200',
                    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
                    'RSI', 'BB_Upper', 'BB_Lower', 'Volatility']

        if 'Sentiment' in processed_df.columns:
            features.append('Sentiment')

    # Get the most recent data
    recent_data = processed_df[features].values[-n_steps:]

    # Scale the data
    recent_data_scaled = scaler_X.transform(recent_data)

    # Reshape for LSTM input
    recent_data_reshaped = recent_data_scaled.reshape(1, n_steps, len(features))

    # Get the latest actual price
    latest_price = processed_df['Close'].iloc[-1]
    latest_date = processed_df['Date'].iloc[-1]
    prediction_date = latest_date + pd.Timedelta(days=1)

    # Print the actual closing price
    print(f"Latest Date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"Latest Closing Price: ${latest_price:.2f}")
    print(f"Prediction for: {prediction_date.strftime('%Y-%m-%d')}")
    print("-" * 50)

    results = {}

    # BNN Model Prediction
    if 'bnn_model' in models_dict:
        bnn_model = models_dict['bnn_model']

        # Make prediction
        prediction = bnn_model.predict(recent_data_reshaped)
        prediction_mean = prediction[0, 0]
        prediction_logvar = prediction[0, 1]

        # Calculate uncertainty
        prediction_std = np.sqrt(np.exp(prediction_logvar))

        # Inverse transform to get actual price
        prediction_mean_inv = scaler_y.inverse_transform([[prediction_mean]])[0][0]

        # Convert numpy array to scalar for string formatting
        prediction_std_inv = float(prediction_std * (scaler_y.data_max_ - scaler_y.data_min_))

        # Calculate confidence interval
        lower_bound = prediction_mean_inv - 2 * prediction_std_inv
        upper_bound = prediction_mean_inv + 2 * prediction_std_inv

        # Calculate the predicted change
        predicted_change = prediction_mean_inv - latest_price
        predicted_change_pct = (predicted_change / latest_price) * 100

        # Print the prediction
        print("Bayesian Neural Network (BNN) Prediction:")
        print(f"Predicted Price: ${prediction_mean_inv:.2f}")
        print(f"Predicted Change: ${predicted_change:.2f} ({predicted_change_pct:.2f}%)")
        print(f"95% Confidence Interval: ${lower_bound:.2f} to ${upper_bound:.2f}")
        print(f"Uncertainty: ${prediction_std_inv:.2f}")
        print("-" * 50)

        results['BNN'] = {
            'prediction': prediction_mean_inv,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'uncertainty': prediction_std_inv,
            'change': predicted_change,
            'change_pct': predicted_change_pct
        }

    # LSTM Model Prediction
    if 'lstm_model' in models_dict:
        lstm_model = models_dict['lstm_model']

        # Make prediction
        lstm_pred = lstm_model.predict(recent_data_reshaped)
        lstm_pred_inv = scaler_y.inverse_transform(lstm_pred)[0][0]

        # Calculate the predicted change
        lstm_predicted_change = lstm_pred_inv - latest_price
        lstm_predicted_change_pct = (lstm_predicted_change / latest_price) * 100

        # Print the prediction
        print("LSTM Model Prediction:")
        print(f"Predicted Price: ${lstm_pred_inv:.2f}")
        print(f"Predicted Change: ${lstm_predicted_change:.2f} ({lstm_predicted_change_pct:.2f}%)")
        print("-" * 50)

        results['LSTM'] = {
            'prediction': lstm_pred_inv,
            'change': lstm_predicted_change,
            'change_pct': lstm_predicted_change_pct
        }

    # Return all prediction results
    return {
        'date': prediction_date,
        'latest_price': latest_price,
        'latest_date': latest_date,
        'models': results
    }

def visualize_todays_predictions(prediction_results, symbol):
    """
    Visualize today's predictions from multiple models
    """
    plt.figure(figsize=(12, 8))

    # Get prediction date and latest price
    prediction_date = prediction_results['date']
    latest_price = prediction_results['latest_price']

    # Set up the x-axis with two dates (latest and prediction)
    dates = [prediction_results['latest_date'], prediction_date]
    date_labels = [d.strftime('%Y-%m-%d') for d in dates]

    # Plot the latest actual price
    plt.plot([0], [latest_price], 'ko', markersize=10, label=f"Latest Close (${latest_price:.2f})")

    # Add horizontal line for reference
    plt.axhline(y=latest_price, color='gray', linestyle='--', alpha=0.5)

    # Plot predictions for each model
    colors = {'BNN': 'red', 'LSTM': 'blue'}
    model_results = prediction_results['models']

    for i, (model_name, results) in enumerate(model_results.items()):
        # Plot prediction point
        plt.plot([1], [results['prediction']], 'o', color=colors[model_name],
                 markersize=10, label=f"{model_name} (${results['prediction']:.2f})")

        # Add uncertainty band for BNN
        if model_name == 'BNN' and 'lower_bound' in results:
            plt.vlines(x=1, ymin=results['lower_bound'], ymax=results['upper_bound'],
                      color=colors[model_name], alpha=0.5, linewidth=8,
                      label=f"BNN 95% Confidence")

    # Set up the plot
    plt.title(f"{symbol} Price Predictions for {prediction_date.strftime('%Y-%m-%d')}", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price ($)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks([0, 1], date_labels, rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('model_comparison_today.png', dpi=300)
    plt.show()

# Update the main function to include predictions from both models
def main():
    # Define parameters
    symbol = "AAPL"
    start_date = "2015-01-01"
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    print(f"===== BAYESIAN NEURAL NETWORK FOR STOCK MARKET FORECASTING =====")
    print(f"Stock Symbol: {symbol}")
    print(f"Data Period: {start_date} to {end_date}")

    # Phase 1: Data Collection & Preprocessing
    stock_data = download_stock_data(symbol, start_date, end_date)
    sentiment_data = fetch_news_sentiment(symbol, start_date, end_date)
    processed_df = preprocess_data(stock_data, sentiment_data)

    # Feature creation
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, features = create_features_target(processed_df)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Features used: {features}")

    # Phase 3: Training & Evaluation
    results, bnn_model, lstm_model = evaluate_models(X_train, X_test, y_train, y_test, scaler_y, processed_df)

    # Convert y_test back to original scale for visualization
    y_test_inv = scaler_y.inverse_transform(y_test)

    # Phase 4: Visualization
    visualize_results(results, y_test_inv)

    # NEW: Get today's predictions using both models
    models_dict = {
        'bnn_model': bnn_model,
        'lstm_model': lstm_model
    }

    today_predictions = predict_today_with_models(
        models_dict, processed_df, scaler_X, scaler_y, n_steps=60, features=features
    )

    # Visualize today's predictions
    visualize_todays_predictions(today_predictions, symbol)

    # Phase 5: Future Forecasting
    last_sequence = X_test[-1]
    future_prices, future_uncertainty = forecast_future(
        bnn_model, last_sequence, scaler_X, scaler_y, days=30
    )

    # Get last actual prices for plotting
    last_actual_prices = processed_df['Close'].values[-100:]

    # Plot forecast
    plot_forecast(last_actual_prices, future_prices, future_uncertainty, symbol)

if __name__ == "__main__":
    main()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
from sklearn.preprocessing import MinMaxScaler
import sqlite3
from datetime import datetime, timedelta
import json
import os

app = FastAPI(title="Stock Prediction API", description="API for predicting stock prices using a Bayesian Neural Network")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define request models
class StockRequest(BaseModel):
    symbol: str
    days: int = 30

class TrainRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str = datetime.now().strftime("%Y-%m-%d")

# Initialize model and scalers
model = None
scaler_X = None
scaler_y = None
features = None

# Helper functions
def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # Simple Moving Averages (SMA)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # Exponential Moving Averages (EMA)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

    # Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=30).std()

    # Moving Average Crossover
    df['MA_Crossover'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)

    # Price to Moving Average Ratio
    df['Price_to_MA_Ratio'] = df['Close'] / df['SMA_50']

    return df

def fetch_stock_data(symbol, start_date, end_date=None):
    """Fetch stock data from Yahoo Finance"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        # Download data
        temp_df = yf.download(symbol, start=start_date, end=end_date)

        # Create a new DataFrame with proper indexing
        df = pd.DataFrame()
        df['Date'] = temp_df.index
        df['Open'] = temp_df['Open'].values
        df['High'] = temp_df['High'].values
        df['Low'] = temp_df['Low'].values
        df['Close'] = temp_df['Close'].values
        df['Volume'] = temp_df['Volume'].values
        df['Adj Close'] = temp_df['Adj Close'].values if 'Adj Close' in temp_df.columns else temp_df['Close'].values

        # Calculate technical indicators
        df = calculate_technical_indicators(df)

        # Drop rows with NaN values
        df.dropna(inplace=True)

        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data: {str(e)}")

def prepare_data(df, n_steps=60):
    """Prepare data for prediction"""
    global features, scaler_X, scaler_y

    # Define features if not defined
    if features is None:
        features = ['Open', 'High', 'Low', 'Volume',
                    'SMA_5', 'SMA_20', 'SMA_50', 'SMA_200',
                    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
                    'RSI', 'BB_Upper', 'BB_Lower', 'Volatility']

    # Create scalers if not defined
    if scaler_X is None:
        scaler_X = MinMaxScaler()
        scaler_X.fit(df[features])

    if scaler_y is None:
        scaler_y = MinMaxScaler()
        scaler_y.fit(df[['Close']])

    # Scale the data
    X = scaler_X.transform(df[features])

    # Create sequence
    X_seq = []
    for i in range(len(X) - n_steps + 1):
        X_seq.append(X[i:i+n_steps])

    X_seq = np.array(X_seq)

    return X_seq, df

def negative_log_likelihood(y_true, y_pred):
    """Custom loss function for probabilistic model"""
    # Extract mean and log variance from the model output
    mean = y_pred[:, 0:1]
    log_var = y_pred[:, 1:2]

    # Calculate negative log likelihood
    return 0.5 * tf.reduce_mean(
        tf.exp(-log_var) * tf.square(y_true - mean) + log_var
    )

def build_model(input_shape):
    """Build a Bayesian LSTM model"""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.InputLayer(input_shape=input_shape),

        # First LSTM layer
        tf.keras.layers.LSTM(units=64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),

        # Second LSTM layer
        tf.keras.layers.LSTM(units=32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),

        # Output layer with mean and log variance
        tf.keras.layers.Dense(2)  # 2 outputs: mean and log_variance
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=negative_log_likelihood
    )

    return model

def save_model(model, symbol):
    """Save the model to disk"""
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save model architecture
    model_json = model.to_json()
    with open(f"models/{symbol}_model.json", "w") as json_file:
        json_file.write(model_json)

    # Save model weights
    model.save_weights(f"models/{symbol}_weights.h5")

    # Save scalers
    np.save(f"models/{symbol}_scaler_X.npy", [scaler_X.data_min_, scaler_X.data_max_])
    np.save(f"models/{symbol}_scaler_y.npy", [scaler_y.data_min_, scaler_y.data_max_])

    # Save features
    with open(f"models/{symbol}_features.json", "w") as f:
        json.dump(features, f)

def load_model_files(symbol):
    """Load model from disk"""
    global model, scaler_X, scaler_y, features

    try:
        # Load model architecture
        with open(f"models/{symbol}_model.json", "r") as json_file:
            loaded_model_json = json_file.read()

        # Create model from JSON
        model = model_from_json(loaded_model_json)

        # Load model weights
        model.load_weights(f"models/{symbol}_weights.h5")

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=negative_log_likelihood
        )

        # Load scalers
        scaler_X_data = np.load(f"models/{symbol}_scaler_X.npy")
        scaler_y_data = np.load(f"models/{symbol}_scaler_y.npy")

        scaler_X = MinMaxScaler()
        scaler_X.data_min_ = scaler_X_data[0]
        scaler_X.data_max_ = scaler_X_data[1]

        scaler_y = MinMaxScaler()
        scaler_y.data_min_ = scaler_y_data[0]
        scaler_y.data_max_ = scaler_y_data[1]

        # Load features
        with open(f"models/{symbol}_features.json", "r") as f:
            features = json.load(f)

        return True
    except Exception as e:
        return False

def forecast_future(model, last_sequence, days=30):
    """Forecast future stock prices"""
    global scaler_X, scaler_y

    # Initialize with the last observed sequence
    current_seq = last_sequence.copy()

    # Store predictions
    future_preds = []
    future_uncertainty = []

    # Make predictions for specified number of days
    for _ in range(days):
        # Get prediction with mean and log variance
        pred = model.predict(current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1]), verbose=0)
        pred_mean = pred[0, 0]
        pred_logvar = pred[0, 1]
        pred_std = np.sqrt(np.exp(pred_logvar))

        # Store prediction and uncertainty
        future_preds.append(pred_mean)
        future_uncertainty.append(pred_std)

        # Update sequence for next prediction
        new_row = np.zeros(current_seq.shape[1])
        new_row[0] = pred_mean  # Assuming first feature is close price

        # Shift sequence and add new prediction
        current_seq = np.vstack([current_seq[1:], new_row])

    # Convert predictions to original scale
    future_preds = np.array(future_preds).reshape(-1, 1)
    future_preds_inv = scaler_y.inverse_transform(future_preds)

    # Scale uncertainty
    future_uncertainty = np.array(future_uncertainty) * (scaler_y.data_max_ - scaler_y.data_min_)

    return future_preds_inv, future_uncertainty

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/train")
def train_model(data: TrainRequest):
    """Train a new model for a stock"""
    global model, scaler_X, scaler_y, features

    try:
        # Fetch data
        df = fetch_stock_data(data.symbol, data.start_date, data.end_date)

        # Define features
        features = ['Open', 'High', 'Low', 'Volume',
                    'SMA_5', 'SMA_20', 'SMA_50', 'SMA_200',
                    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
                    'RSI', 'BB_Upper', 'BB_Lower', 'Volatility']

        # Create scalers
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        # Scale the data
        X = scaler_X.fit_transform(df[features])
        y = scaler_y.fit_transform(df[['Close']])

        # Create sequences for LSTM
        X_seq, y_seq = [], []
        n_steps = 60  # Number of time steps to use

        for i in range(len(X) - n_steps):
            X_seq.append(X[i:i+n_steps])
            y_seq.append(y[i+n_steps])

        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        # Build and train model
        model = build_model(input_shape=(X_seq.shape[1], X_seq.shape[2]))

        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        # Train model
        history = model.fit(
            X_seq, y_seq,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        # Save model
        save_model(model, data.symbol)

        return {
            "message": f"Model trained successfully for {data.symbol}",
            "data_points": len(df),
            "training_samples": len(X_seq),
            "features": features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.post("/predict")
def predict_stock(data: StockRequest):
    """Predict stock prices"""
    global model, scaler_X, scaler_y, features

    try:
        # Check if model exists
        if not load_model_files(data.symbol):
            # If model doesn't exist, return error
            return {
                "error": f"No model found for {data.symbol}. Please train a model first.",
                "status": "error"
            }

        # Fetch recent data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        df = fetch_stock_data(data.symbol, start_date, end_date)

        # Prepare data
        X_seq, df = prepare_data(df)

        # Get last sequence
        last_sequence = X_seq[-1]

        # Forecast future prices
        future_prices, future_uncertainty = forecast_future(model, last_sequence, days=data.days)

        # Get recent actual prices
        recent_prices = df['Close'].values[-30:].tolist()
        recent_dates = df['Date'].dt.strftime('%Y-%m-%d').values[-30:].tolist()

        # Generate future dates
        last_date = df['Date'].iloc[-1]
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(data.days)]

        # Get technical indicators
        technical_indicators = {}
        for indicator in ['RSI', 'MACD', 'SMA_50', 'SMA_200']:
            if indicator in df.columns:
                technical_indicators[indicator] = float(df[indicator].iloc[-1])

        # Calculate current trend
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100

        # Determine trend
        if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
            trend = "Bullish"
        else:
            trend = "Bearish"

        # Prepare response
        response = {
            "symbol": data.symbol,
            "last_price": float(current_price),
            "price_change": float(price_change),
            "price_change_pct": float(price_change_pct),
            "trend": trend,
            "technical_indicators": technical_indicators,
            "recent_prices": {
                "dates": recent_dates,
                "prices": [float(p) for p in recent_prices]
            },
            "forecast": {
                "dates": future_dates,
                "prices": [float(p[0]) for p in future_prices],
                "lower_bound": [float(p[0] - 2*u) for p, u in zip(future_prices, future_uncertainty)],
                "upper_bound": [float(p[0] + 2*u) for p, u in zip(future_prices, future_uncertainty)]
            }
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.get("/symbols")
def get_symbols():
    """Get list of available symbols with trained models"""
    if not os.path.exists('models'):
        return {"symbols": []}

    symbols = []
    for file in os.listdir('models'):
        if file.endswith('_model.json'):
            symbols.append(file.split('_')[0])

    return {"symbols": list(set(symbols))}

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)