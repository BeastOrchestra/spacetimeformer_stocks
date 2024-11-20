import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import mpld3
from ib_insync import IB, Stock, util
from pandas.tseries.offsets import BDay

# File paths
tickers_file = "tickers.txt"
predictions_file = "/Users/alecjeffery/Documents/Playgrounds/Python/spacetimeformer_stocks/oos_predictions_current.csv"
output_file = "all_plots.html"

# Initialize IB connection
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=11)

# List to store HTML content of each plot
html_plots = []

# Function to fetch adjusted close price and implied volatility data
def get_historical_data_with_volatility(ticker):
    contract = Stock(ticker, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    # Fetch adjusted close price data
    historical_data = ib.reqHistoricalData(
        contract,
        endDateTime='',
        barSizeSetting='1 day',
        durationStr='6 M',
        whatToShow='ADJUSTED_LAST',
        useRTH=True
    )
    price_df = util.df(historical_data)
    price_df.set_index('date', inplace=True)

    # Fetch implied volatility data
    iv_data = ib.reqHistoricalData(
        contract,
        endDateTime='',
        barSizeSetting='1 day',
        durationStr='6 M',
        whatToShow='OPTION_IMPLIED_VOLATILITY',
        useRTH=True
    )
    iv_df = util.df(iv_data)
    iv_df.set_index('date', inplace=True)

    # Rename and merge volatility columns
    iv_df = iv_df.rename(columns={'open': 'vopen', 'high': 'vhigh', 'low': 'vlow', 'close': 'vclose'})
    merged_df = price_df.join(iv_df[['vopen', 'vhigh', 'vlow', 'vclose']], how='left')

    # Drop unnecessary columns
    merged_df = merged_df.drop(columns=['average', 'barCount'], errors='ignore')
    return merged_df

# Function to calculate percentage change
def calculate_percentage_change(current, forecast):
    return ((forecast / current) - 1) * 100 if current != 0 else 0

# Updated function to plot stock data with candlestick chart and line plot for volatility
def plot_stock_predictions(ticker, position_category, actual_data, predicted_data):
    global html_plots

    # Calculate 50-day and 200-day SMAs
    actual_data['SMA_50'] = actual_data['close'].rolling(window=50).mean()
    actual_data['SMA_200'] = actual_data['close'].rolling(window=200).mean()

    # Prepare candlestick data for price
    ohlc_data = actual_data[['open', 'high', 'low', 'close']].copy()
    ohlc_data.index = pd.to_datetime(ohlc_data.index)

    # Extract predicted close and volatility values
    predicted_close_values = predicted_data[['Close_1', 'Close_2', 'Close_3', 'Close_4', 'Close_5',
                                             'Close_6', 'Close_7', 'Close_8', 'Close_9', 'Close_10']].values.flatten()
    predicted_volatility_values = predicted_data[['Volatility_1', 'Volatility_2', 'Volatility_3', 'Volatility_4',
                                                  'Volatility_5', 'Volatility_6', 'Volatility_7', 'Volatility_8',
                                                  'Volatility_9', 'Volatility_10']].values.flatten()

    prediction_dates = pd.date_range(start=ohlc_data.index[-1], periods=11, freq=BDay())[1:]  # 10 business days

    # Extract the final predicted values
    final_close_10 = predicted_data['Close_10'].values[0]
    final_volatility_10 = predicted_data['Volatility_10'].values[0]

    # Calculate percentage changes
    last_actual_close = actual_data['close'].iloc[-1]
    last_actual_volatility = actual_data['vclose'].iloc[-1]

    close_pct_change = calculate_percentage_change(last_actual_close, final_close_10)
    volatility_pct_change = calculate_percentage_change(last_actual_volatility, final_volatility_10)

    # Create figure for candlestick and line plots
    fig, axs = plt.subplots(2, 1, figsize=(7, 12))
    fontdict = {'fontsize': 16}

    # Plot candlestick chart for stock prices
    mpf.plot(
        ohlc_data,
        type='candle',
        ax=axs[0],
        style='yahoo',
        mav=(50, 200),
        ylabel='Price ($)',
        volume=False,
        show_nontrading=True
    )
    axs[0].plot(prediction_dates, predicted_close_values, color='red', linestyle='--', label='Predicted Close Prices')
    axs[0].set_title(
        f'{ticker} Close Prices & Forecast: ${final_close_10:.2f} [{close_pct_change:.2f}%]\nPosition: {position_category.upper()}',
        fontdict=fontdict
    )
    axs[0].legend()
    axs[0].grid()

    # Plot line chart for implied volatility
    actual_dates = pd.to_datetime(actual_data.index)
    axs[1].plot(actual_dates, actual_data['vclose'], label=f'{ticker} Volatility (IV)', color='blue')
    axs[1].plot(prediction_dates, predicted_volatility_values, label='Predicted Volatility (IV)', color='red', linestyle='--')
    axs[1].set_title(
        f'{ticker} Volatility & Forecast: {final_volatility_10:.2f}% [{volatility_pct_change:.2f}%]\nPosition: {position_category.upper()}',
        fontdict=fontdict
    )
    axs[1].set_xlabel('Date', fontsize=12)
    axs[1].set_ylabel('Volatility (%)', fontsize=16)
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].grid(axis='y')
    axs[1].legend(fontsize=16)

    plt.tight_layout()

    # Convert the plot to HTML using mpld3
    html_str = mpld3.fig_to_html(fig)
    html_plots.append(html_str)
    print(f"Added candlestick and volatility line plot for {ticker} with position '{position_category}'.")


# Function to save all plots into a single HTML file
def save_combined_html(output_file):
    with open(output_file, 'w') as file:
        file.write("<html><head><title>Stock Predictions</title></head><body>\n")
        for html_str in html_plots:
            file.write(html_str)
            file.write("<hr>\n")
        file.write("</body></html>")
    print(f"Saved all plots to {output_file}")

# Main function to process tickers and generate plots
def process_tickers():
    # Read ticker symbols from tickers.txt
    with open(tickers_file, 'r') as file:
        tickers = [line.strip().split(',') for line in file]

    # Load unscaled out-of-sample predictions
    predictions_df = pd.read_csv(predictions_file)

    for ticker, position_category in tickers:
        ticker = ticker.upper()
        position_category = position_category.lower()

        # Fetch unscaled historical data with volatility
        try:
            print(f"Fetching data for {ticker}...")
            actual_data = get_historical_data_with_volatility(ticker)

            # Filter predictions for the current ticker
            prediction_data = predictions_df[predictions_df['Unnamed: 0'] == ticker]
            if prediction_data.empty:
                print(f"No prediction data found for {ticker}. Skipping.")
                continue

            # Plot stock predictions with candlestick and line plots
            plot_stock_predictions(ticker, position_category, actual_data, prediction_data)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Disconnect from IB
    ib.disconnect()

# Execute the main function and save the HTML
if __name__ == "__main__":
    process_tickers()
    save_combined_html(output_file)

