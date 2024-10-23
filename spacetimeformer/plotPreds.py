import os
import pandas as pd
import matplotlib.pyplot as plt

def load_data(stock_name):
    # Load recent stock data from the oos folder
    stock_data_path = f'spacetimeformer/data/oos/{stock_name}.csv'
    if os.path.exists(stock_data_path):
        stock_data = pd.read_csv(stock_data_path)
        return stock_data
    else:
        raise FileNotFoundError(f"No data found for {stock_name} in the specified path.")

def load_predictions():
    # Load predictions from the latest oos_predictions file
    predictions_files = [f for f in os.listdir() if f.startswith('oos_predictions_') and f.endswith('.csv')]
    if predictions_files:
        latest_file = max(predictions_files, key=os.path.getctime)
        predictions = pd.read_csv(latest_file, index_col=0)
        return predictions
    else:
        raise FileNotFoundError("No predictions file found.")

def plot_stock_and_predictions(stock_data, predictions, stock_name):
    plt.figure(figsize=(14, 7))
    
    # Plot recent stock history
    plt.plot(stock_data['Datetime'], stock_data['Close'], label='Actual Close Price', color='blue')
    
    # Plot predictions
    plt.plot(predictions.index, predictions['Close_10'], label='Predicted Close Price', color='orange', linestyle='--')
    
    plt.title(f'Stock Price and Predictions for {stock_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main(stock_name):
    stock_data = load_data(stock_name)
    predictions = load_predictions()
    
    # Calculate Price and Volatility deltas
    predictions['Price_PrctDelta'] = predictions['Close_10'] - predictions['Close_1']
    predictions['Volatility_PrctDelta'] = predictions['Volatility_10'] - predictions['Volatility_1']
    
    # Filter opportunities
    PossibleLongCalls = predictions[(predictions['Price_PrctDelta'] > 0) & (predictions['Volatility_PrctDelta'] > 0)]
    PossibleLongPuts = predictions[(predictions['Price_PrctDelta'] < 0) & (predictions['Volatility_PrctDelta'] > 0)]
    
    # Print opportunities
    print('Possible Long Calls:', PossibleLongCalls)
    print('Possible Long Puts:', PossibleLongPuts)
    
    # Plot the stock data and predictions
    plot_stock_and_predictions(stock_data, predictions, stock_name)

if __name__ == "__main__":
    stock_name = input("Enter the stock name (without .csv): ")
    main(stock_name)
