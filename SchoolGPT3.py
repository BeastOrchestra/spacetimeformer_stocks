import os
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from mplchart.chart import Chart
from mplchart.primitives import Candlesticks, Volume
from mplchart.indicators import SMA, RSI, MACD
from matplotlib.backends.backend_pdf import PdfPages
from ib_insync import IB, Stock, util
from pandas.tseries.offsets import BDay

# File paths
tickers_file = "tickers.txt"
predictions_file = "/Users/alecjeffery/Documents/Playgrounds/Python/spacetimeformer_stocks/oos_predictions_current.csv"
output_pdf = "all_charts.pdf"

# Initialize IB connection
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=11)

# Function to fetch adjusted close price and implied volatility data
def get_historical_data(ticker):
    contract = Stock(ticker, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    historical_data = ib.reqHistoricalData(
        contract,
        endDateTime='',
        barSizeSetting='1 day',
        durationStr='2 Y',
        whatToShow='ADJUSTED_LAST',
        useRTH=True
    )
    data = util.df(historical_data)
    data.set_index('date', inplace=True)

    try:
        iv_data = ib.reqHistoricalData(
            contract,
            endDateTime='',
            barSizeSetting='1 day',
            durationStr='2 Y',
            whatToShow='OPTION_IMPLIED_VOLATILITY',
            useRTH=True
        )
        iv_df = util.df(iv_data)
        iv_df.set_index('date', inplace=True)
        iv_df = iv_df.rename(columns={'close': 'vclose'})
        data = data.join(iv_df[['vclose']], how='left')
    except Exception:
        print(f"Warning: No implied volatility data found for {ticker}. Setting vclose to NaN.")
        data['vclose'] = float('nan')

    return data

# Function to calculate Ichimoku Cloud components
def calculate_ichimoku(data):
    """
    Calculate Ichimoku Cloud components.
    """
    nine_period_high = data['high'].rolling(window=9).max()
    nine_period_low = data['low'].rolling(window=9).min()
    data['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    twenty_six_period_high = data['high'].rolling(window=26).max()
    twenty_six_period_low = data['low'].rolling(window=26).min()
    data['kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2

    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
    fifty_two_period_high = data['high'].rolling(window=52).max()
    fifty_two_period_low = data['low'].rolling(window=52).min()
    data['senkou_span_b'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)

    data['chikou_span'] = data['close'].shift(-26)
    return data

# Function to plot Ichimoku Cloud chart
def plot_ichimoku(ticker, data):
    """
    Plot Ichimoku Cloud chart using mplfinance.
    """
    ichimoku_cloud = [
        mpf.make_addplot(data['tenkan_sen'], color='blue', width=1),
        mpf.make_addplot(data['kijun_sen'], color='red', width=1),
        mpf.make_addplot(data['senkou_span_a'], color='lightgreen', width=0.5),
        mpf.make_addplot(data['senkou_span_b'], color='lightcoral', width=0.5),
        mpf.make_addplot(data['chikou_span'], color='green', width=1)
    ]

    fill_up = dict(
        y1=data['senkou_span_a'].values,
        y2=data['senkou_span_b'].values,
        where=data['senkou_span_a'] >= data['senkou_span_b'],
        alpha=0.5,
        color='honeydew'
    )
    fill_down = dict(
        y1=data['senkou_span_a'].values,
        y2=data['senkou_span_b'].values,
        where=data['senkou_span_a'] < data['senkou_span_b'],
        alpha=0.5,
        color='mistyrose'
    )

    mpf.plot(
        data,
        type='candle',
        addplot=ichimoku_cloud,
        fill_between=[fill_up, fill_down],
        title=f"{ticker} Ichimoku Cloud",
        style='yahoo'
    )

# Main function to process tickers and save charts into a single PDF
def process_tickers_to_pdf():
    pdf_pages = PdfPages(output_pdf)

    with open(tickers_file, 'r') as file:
        tickers = [line.strip().split(',') for line in file]

    predictions_df = pd.read_csv(predictions_file)

    for ticker_info in tickers:
        ticker = ticker_info[0].strip().upper()
        position_category = ticker_info[1].strip().lower()

        try:
            print(f"Fetching data for {ticker}...")
            actual_data = get_historical_data(ticker)
            prediction_data = predictions_df[predictions_df['Unnamed: 0'] == ticker]

            if prediction_data.empty:
                print(f"No prediction data found for {ticker}. Skipping.")
                continue

            # Calculate Ichimoku Cloud
            actual_data = calculate_ichimoku(actual_data)

            # Create Ichimoku Chart
            print(f"Creating Ichimoku chart for {ticker}...")
            ichimoku_fig = plt.figure()
            plot_ichimoku(ticker, actual_data)
            pdf_pages.savefig(ichimoku_fig)
            plt.close(ichimoku_fig)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    pdf_pages.close()
    print(f"Saved all charts to {output_pdf}")

if __name__ == "__main__":
    process_tickers_to_pdf()
