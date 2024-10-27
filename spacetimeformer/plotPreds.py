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
    # Load predictions from the specified oos_predictions file
    predictions_path = '/Users/alecjeffery/Documents/Playgrounds/Python/spacetimeformer_stocks/oos_predictions.csv'
    if os.path.exists(predictions_path):
        predictions = pd.read_csv(predictions_path)
        return predictions
    else:
        raise FileNotFoundError("No predictions file found.")

def plot_stock_and_predictions(stock_data, predictions, stock_name):
    plt.figure(figsize=(14, 7))
    
    # Plot recent stock history
    plt.plot(stock_data['Close'], label='Actual Close Price', color='blue')
    
    # Extract the projected values
    projected_values = predictions[['Close_1', 'Close_2', 'Close_3', 'Close_4', 'Close_5', 'Close_6', 'Close_7', 'Close_8', 'Close_9', 'Close_10']].values.flatten()
    
    # Create an index for the projected values
    projection_index = range(len(stock_data), len(stock_data) + len(projected_values))
    
    # Plot predictions
    plt.plot(projection_index, projected_values, label='Predicted Close Price', color='yellow', linestyle='--')
    
    plt.title(f'Stock Price and Predictions for {stock_name}')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main(stock_name):
    stock_data = load_data(stock_name)
    predictions = load_predictions()
    
    # Plot the stock data and predictions
    plot_stock_and_predictions(stock_data, predictions, stock_name)

if __name__ == "__main__":
    stock_name = 'ADBE'  # Hardcoding ADBE as per the user's request
    main(stock_name)
