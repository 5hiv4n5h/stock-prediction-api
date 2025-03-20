from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime

app = Flask(__name__)

from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Loss
import tensorflow as tf

# Define the custom loss function again
def negative_log_likelihood(y_true, y_pred):
    mean = y_pred[:, 0:1]
    log_var = y_pred[:, 1:2]
    return 0.5 * tf.reduce_mean(tf.exp(-log_var) * tf.square(y_true - mean) + log_var)

# Load model with custom loss
model = load_model("bnn_model.h5", custom_objects={"negative_log_likelihood": negative_log_likelihood})

# Load scalers (if used)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance and process it."""
    try:
        # Fetch stock data
        df = yf.download(symbol, start=start_date, end=end_date)
        
        print(f"Downloaded data for {symbol}: {df.shape} rows")
        print("DataFrame columns:", df.columns.tolist())
        
        if df.empty:
            print(f"❌ Error: No stock data returned for {symbol}")
            return None
            
        # Handle yfinance's output format (it might return MultiIndex columns)
        if isinstance(df.columns, pd.MultiIndex):
            print("Detected MultiIndex columns, flattening...")
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Check if the required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {df.columns.tolist()}")
            return None
            
        # Select only the needed columns to prevent errors
        df = df[required_columns].copy()
        
        # Reset index to make Date a column
        df = df.reset_index()
        print("Columns after reset_index:", df.columns.tolist())
        
        # Create technical indicators based on available data
        # Use smaller windows if we have limited data
        min_window = min(5, len(df) - 1) if len(df) < 60 else 5
        df['SMA_5'] = df['Close'].rolling(window=min_window).mean()
        
        if len(df) >= 20:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['BB_Upper'] = df['Close'].rolling(window=20).mean() + (df['Close'].rolling(window=20).std() * 2)
            df['BB_Lower'] = df['Close'].rolling(window=20).mean() - (df['Close'].rolling(window=20).std() * 2)
        else:
            # Use smaller windows for limited data
            small_window = max(3, len(df) // 3)
            df['SMA_20'] = df['Close'].rolling(window=small_window).mean()
            df['BB_Upper'] = df['Close'].rolling(window=small_window).mean() + (df['Close'].rolling(window=small_window).std() * 2)
            df['BB_Lower'] = df['Close'].rolling(window=small_window).mean() - (df['Close'].rolling(window=small_window).std() * 2)
        
        if len(df) >= 50:
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
        else:
            # Use smaller window for SMA_50
            df['SMA_50'] = df['Close'].rolling(window=min(len(df) // 2, 10)).mean()
            
        if len(df) >= 200:
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
        else:
            # Skip SMA_200 or use a smaller substitute
            df['SMA_200'] = df['Close'].rolling(window=min(len(df) // 2, 20)).mean()
            
        # MACD with adaptive windows
        ema_window1 = min(12, max(3, len(df) // 10))
        ema_window2 = min(26, max(6, len(df) // 5))
        df['EMA_12'] = df['Close'].ewm(span=ema_window1, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=ema_window2, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=min(9, max(3, len(df) // 15)), adjust=False).mean()
        
        # Calculate RSI safely with adaptive window
        rsi_window = min(14, max(5, len(df) // 10))
        delta = df['Close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        avg_gain = gain.rolling(window=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate volatility with adaptive window
        vol_window = min(30, max(5, len(df) // 5))
        df['Volatility'] = df['Close'].pct_change().rolling(window=vol_window).std()
        
        # Drop the Date column before returning
        date_column = df.columns[0] if 'Date' in df.columns[0] else 'Date'
        df = df.drop(columns=[date_column])
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        print(f"Processed data shape: {df.shape}")
        print(f"Final columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error in fetch_stock_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.route("/")
def home():
    return jsonify({"message": "Stock Prediction API is Running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        symbol = data.get("symbol", "AAPL")  # Default to AAPL
        days = int(data.get("days", 30))  # Default to 30 days
        
        # Fetch recent stock data with extended date range
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        # Use 3 years of data instead of 1 year to ensure we have enough data points
        start_date = (datetime.datetime.now() - datetime.timedelta(days=3*365)).strftime("%Y-%m-%d")
        
        print(f"Fetching data for {symbol} from {start_date} to {end_date}")
        df = fetch_stock_data(symbol, start_date, end_date)
        
        # Check if data is available
        if df is None or df.empty:
            return jsonify({"error": f"No data available for symbol {symbol}"}), 400
            
        # Reduce minimum required rows from 60 to 30
        min_required_rows = 30
        if len(df) < min_required_rows:
            return jsonify({"error": f"Not enough data available for prediction. Need at least {min_required_rows} days of data."}), 400
            
        # Print shape information for debugging
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Preprocess the data
        feature_count = df.shape[1]
        sequence_length = min(60, len(df))  # Use all available data up to 60 days
        
        # Use last available data points
        X = scaler_X.fit_transform(df.values[-sequence_length:])
        print("X shape before reshaping:", X.shape)
        
        X = X.reshape(1, sequence_length, X.shape[1])  # Reshape for LSTM
        print("X shape after reshaping:", X.shape)
        
        # Make prediction
        pred = model.predict(X)
        
        # Ensure prediction is properly shaped for inverse transform
        pred_reshaped = pred[:, 0].reshape(-1, 1) if pred.shape[1] > 1 else pred
        
        # Initialize scaler_y with some reasonable values if it wasn't fit
        # This is a workaround and should be adjusted based on your model's output range
        if not hasattr(scaler_y, 'scale_') or scaler_y.scale_ is None:
            print("Warning: scaler_y was not fitted. Using a default scaling.")
            # Use recent closing prices to fit the scaler for a reasonable output
            recent_closes = df['Close'].values[-min_required_rows:].reshape(-1, 1)
            scaler_y.fit(recent_closes)
        
        pred_price = scaler_y.inverse_transform(pred_reshaped)[0][0]
        
        # Get uncertainty if available (for Bayesian models)
        uncertainty = None
        if pred.shape[1] > 1:
            uncertainty = np.exp(pred[0, 1])  # For log variance
            
        response = {
            "symbol": symbol,
            "predicted_price": round(float(pred_price), 2),
            "date": end_date,
            "data_points_used": sequence_length
        }
        
        if uncertainty is not None:
            response["uncertainty"] = round(float(uncertainty), 4)
            
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)