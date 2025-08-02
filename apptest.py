from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from datetime import datetime, timedelta
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Load model and scaler
try:
    os.makedirs('stock', exist_ok=True)
    model = load_model('stock/model.h5')
    scaler = joblib.load('stock/scaler.pkl')
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Enhanced HTML template with tables and multiple graphs
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Stock Price Prediction Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        input, button { padding: 12px; margin: 10px 0; width: 100%; border: 1px solid #ddd; border-radius: 5px; }
        button { background-color: #007bff; color: white; border: none; cursor: pointer; font-weight: bold; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 8px; }
        .error { color: red; background-color: #fee; }
        .success { color: green; }
        .graph { text-align: center; margin: 30px 0; }
        .graph img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .metrics { display: flex; justify-content: space-around; background: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .metric-box { text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: center; border: 1px solid #ddd; }
        th { background-color: #007bff; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .table-container { max-height: 400px; overflow-y: auto; }
        h3 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Stock Price Prediction Dashboard</h1>
        <form id="predictionForm">
            <input type="text" id="stockId" placeholder="Enter Stock Symbol (e.g., AAPL, TSLA, MSFT)" required>
            <button type="submit">🔍 Generate Complete Analysis</button>
        </form>
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const stockId = document.getElementById('stockId').value;
            const resultDiv = document.getElementById('result');
            
            resultDiv.innerHTML = '<div class="result"><p>🔄 Loading prediction, generating graphs, and creating tables...</p></div>';
            
            try {
                const response = await fetch('/complete_analysis', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({stock_id: stockId})
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    resultDiv.innerHTML = `
                        <div class="result success">
                            <h3>📊 Analysis Results for ${data.stock}</h3>
                            
                            <div class="metrics">
                                <div class="metric-box">
                                    <div>Current Price</div>
                                    <div class="metric-value">$${data.current_price.toFixed(2)}</div>
                                </div>
                                <div class="metric-box">
                                    <div>Next Day Prediction</div>
                                    <div class="metric-value">$${data.predicted_price.toFixed(2)}</div>
                                </div>
                                <div class="metric-box">
                                    <div>RMSE</div>
                                    <div class="metric-value">${data.rmse.toFixed(2)}</div>
                                </div>
                                <div class="metric-box">
                                    <div>Prediction Change</div>
                                    <div class="metric-value" style="color: ${data.price_change >= 0 ? 'green' : 'red'}">
                                        ${data.price_change >= 0 ? '↗️' : '↘️'} ${data.price_change.toFixed(2)}%
                                    </div>
                                </div>
                            </div>

                            <h3>📈 Original vs Predicted Prices</h3>
                            <div class="graph">
                                <img src="data:image/png;base64,${data.comparison_graph}" alt="Original vs Predicted">
                            </div>

                            <h3>📊 Close Price with Moving Averages</h3>
                            <div class="graph">
                                <img src="data:image/png;base64,${data.ma_graph}" alt="Close Price with Moving Averages">
                            </div>

                            <h3>🔍 Price Analysis Table</h3>
                            <div class="table-container">
                                ${data.price_table}
                            </div>

                            <h3>📋 Prediction vs Original Values (Last 20 Days)</h3>
                            <div class="table-container">
                                ${data.prediction_table}
                            </div>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `<div class="result error"><p>❌ Error: ${data.error}</p></div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error"><p>❌ Error: ${error.message}</p></div>`;
            }
        });
    </script>
</body>
</html>
"""

def create_comparison_graph(df, predictions, actual_prices, stock_id):
    """Create comparison graph of original vs predicted prices"""
    plt.figure(figsize=(14, 8))
    
    dates = df.index[-len(actual_prices):]
    
    plt.plot(dates, actual_prices.flatten(), label='Original Price', color='blue', linewidth=2)
    plt.plot(dates, predictions.flatten(), label='Predicted Price', color='red', linewidth=2, linestyle='--')
    
    plt.title(f'{stock_id} Stock Price - Original vs Predicted', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    return img_base64

def create_ma_graph(df, stock_id):
    """Create Close price with Moving Averages graph"""
    plt.figure(figsize=(14, 8))
    
    # Calculate moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    
    # Plot last 2 years of data
    recent_data = df.tail(500)
    
    plt.plot(recent_data.index, recent_data['Close'], label='Close Price', color='black', linewidth=2)
    plt.plot(recent_data.index, recent_data['MA20'], label='MA20', color='orange', alpha=0.7)
    plt.plot(recent_data.index, recent_data['MA50'], label='MA50', color='green', alpha=0.7)
    plt.plot(recent_data.index, recent_data['MA100'], label='MA100', color='red', alpha=0.7)
    
    plt.title(f'{stock_id} Close Price with Moving Averages', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    return img_base64

def create_price_analysis_table(df):
    """Create price analysis table with key metrics"""
    recent_data = df.tail(10).copy()
    
    # Calculate additional metrics
    recent_data['Daily_Change'] = recent_data['Close'].pct_change() * 100
    recent_data['Volume_MA'] = recent_data['Volume'].rolling(window=5).mean()
    recent_data['Price_Range'] = recent_data['High'] - recent_data['Low']
    
    # Create HTML table
    table_html = """
    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Open ($)</th>
                <th>High ($)</th>
                <th>Low ($)</th>
                <th>Close ($)</th>
                <th>Volume</th>
                <th>Daily Change (%)</th>
                <th>Price Range ($)</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for date, row in recent_data.iterrows():
        change_color = 'green' if row['Daily_Change'] >= 0 else 'red'
        table_html += f"""
            <tr>
                <td>{date.strftime('%Y-%m-%d')}</td>
                <td>{row['Open']:.2f}</td>
                <td>{row['High']:.2f}</td>
                <td>{row['Low']:.2f}</td>
                <td><strong>{row['Close']:.2f}</strong></td>
                <td>{int(row['Volume']):,}</td>
                <td style="color: {change_color}"><strong>{row['Daily_Change']:.2f}%</strong></td>
                <td>{row['Price_Range']:.2f}</td>
            </tr>
        """
    
    table_html += "</tbody></table>"
    return table_html

def create_prediction_table(df, predictions, actual_prices):
    """Create prediction vs original table"""
    dates = df.index[-len(actual_prices):][-20:]  # Last 20 days
    pred_last_20 = predictions[-20:]
    actual_last_20 = actual_prices[-20:]
    
    table_html = """
    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Original Price ($)</th>
                <th>Predicted Price ($)</th>
                <th>Difference ($)</th>
                <th>Error (%)</th>
                <th>Accuracy</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for i, date in enumerate(dates):
        original = actual_last_20[i][0]
        predicted = pred_last_20[i][0]
        difference = predicted - original
        error_pct = abs(difference / original) * 100
        accuracy = max(0, 100 - error_pct)
        
        diff_color = 'green' if difference >= 0 else 'red'
        accuracy_color = 'green' if accuracy >= 95 else 'orange' if accuracy >= 90 else 'red'
        
        table_html += f"""
            <tr>
                <td>{date.strftime('%Y-%m-%d')}</td>
                <td><strong>{original:.2f}</strong></td>
                <td><strong>{predicted:.2f}</strong></td>
                <td style="color: {diff_color}"><strong>{difference:+.2f}</strong></td>
                <td>{error_pct:.2f}%</td>
                <td style="color: {accuracy_color}"><strong>{accuracy:.1f}%</strong></td>
            </tr>
        """
    
    table_html += "</tbody></table>"
    return table_html

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/complete_analysis", methods=["POST"])
def complete_analysis():
    """Complete analysis endpoint with graphs and tables"""
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
        
    stock_id = data.get("stock_id", "AAPL").upper()
    
    try:
        # Download historical data
        end = datetime.now()
        start = end - timedelta(days=365 * 5)
        df = yf.download(stock_id, start=start, end=end, progress=False)
        
        # FIX 1: Proper DataFrame empty check
        if df.empty or len(df) == 0:  # Use .empty property, not direct boolean
            return jsonify({"error": f"Stock symbol '{stock_id}' not found"}), 404
        
        # FIX 2: Handle potential MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)  # Remove ticker level
        
        # FIX 3: Ensure we have the right column
        if 'Close' not in df.columns:
            return jsonify({"error": "Close price data not available"}), 400
            
        # Prepare data for prediction
        close_data = df[['Close']].values
        
        # FIX 4: Add proper length checks
        if len(close_data) < 100:
            return jsonify({"error": "Insufficient historical data"}), 400
        
        # Scale the data
        scaled_data = scaler.transform(close_data)
        
        # Create test sequences
        sequence_length = 100
        x_test = []
        y_test = []
        
        test_start = int(len(scaled_data) * 0.7)
        
        for i in range(test_start + sequence_length, len(scaled_data)):
            x_test.append(scaled_data[i-sequence_length:i])
            y_test.append(scaled_data[i])
        
        # FIX 5: Better array handling
        if len(x_test) == 0:
            return jsonify({"error": "Not enough data for prediction"}), 400
        
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        # Make predictions
        predictions = model.predict(x_test, verbose=0)
        
        # Inverse transform
        inv_predictions = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(inv_y_test, inv_predictions))
        current_price = float(close_data[-1][0])  # Ensure it's a float
        
        # Predict next day
        last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
        next_price_scaled = model.predict(last_sequence, verbose=0)
        next_price = float(scaler.inverse_transform(next_price_scaled)[0][0])
        
        # Calculate price change percentage
        price_change = ((next_price - current_price) / current_price) * 100
        
        # Generate graphs and tables with error handling
        try:
            comparison_graph = create_comparison_graph(df, inv_predictions, inv_y_test, stock_id)
            ma_graph = create_ma_graph(df.copy(), stock_id)
            price_table = create_price_analysis_table(df)
            prediction_table = create_prediction_table(df, inv_predictions, inv_y_test)
        except Exception as graph_error:
            return jsonify({"error": f"Graph generation failed: {str(graph_error)}"}), 500
        
        return jsonify({
            "stock": stock_id,
            "predicted_price": float(next_price),
            "current_price": float(current_price),
            "rmse": float(rmse),
            "price_change": float(price_change),
            "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "comparison_graph": comparison_graph,
            "ma_graph": ma_graph,
            "price_table": price_table,
            "prediction_table": prediction_table,
            "test_data_points": len(inv_predictions)
        })
        
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
 

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)

