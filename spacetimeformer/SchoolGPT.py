import os
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions data
predictions_path = '/Users/alecjeffery/Documents/Playgrounds/Python/spacetimeformer_stocks/oos_predictions.csv'
df1 = pd.read_csv(predictions_path)

# Load historical stock data for ADBE
adbe_data_path = 'spacetimeformer/data/oos/ADBE.csv'
adbe_data = pd.read_csv(adbe_data_path)

# Extract Close and v_close values
close_values = adbe_data['Close'].values
v_close_values = adbe_data['vclose'].values

# Extract Close_1 to Close_10 values for ADBE
adbe_prediction_data = df1[df1['Unnamed: 0'] == 'ADBE']
predicted_close_values = adbe_prediction_data.iloc[0][['Close_1', 'Close_2', 'Close_3', 'Close_4', 'Close_5', 'Close_6', 'Close_7', 'Close_8', 'Close_9', 'Close_10']].values
predicted_volatility_values = adbe_prediction_data.iloc[0][['Volatility_1', 'Volatility_2', 'Volatility_3', 'Volatility_4', 'Volatility_5', 'Volatility_6', 'Volatility_7', 'Volatility_8', 'Volatility_9', 'Volatility_10']].values

# Load final forecast values from the specified file
final_forecast_path = '/Users/alecjeffery/Documents/Playgrounds/Python/spacetimeformer_stocks/oos_predictions_10_22_2024.csv'
final_forecast_data = pd.read_csv(final_forecast_path)
final_close_10 = final_forecast_data.loc[final_forecast_data['Unnamed: 0'] == 'ADBE', 'Close_10'].values[0]
final_volatility_10 = final_forecast_data.loc[final_forecast_data['Unnamed: 0'] == 'ADBE', 'Volatility_10'].values[0]

# Create an index for the x-axis
adbe_days = range(len(close_values))  # Index for ADBE Close prices
prediction_days = range(len(close_values), len(close_values) + len(predicted_close_values))  # Index for predictions

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot Close prices
axs[0].plot(adbe_days, close_values, label='ADBE Close Prices', color='blue')
axs[0].plot(prediction_days, predicted_close_values, label='Predicted Close Prices', color='red', linestyle='--')
axs[0].set_title(f'ADBE Close Prices and Predictions\nFinal Close Prediction: ${final_close_10:.2f}', fontsize=12)
axs[0].set_ylabel('Price ($)')
axs[0].grid(axis='y')
axs[0].legend()

# Plot v_close (volatility)
axs[1].plot(adbe_days, v_close_values, label='ADBE Volatility (IV)', color='blue')
axs[1].plot(prediction_days, predicted_volatility_values, label='Predicted Volatility (IV)', color='red', linestyle='--')
axs[1].set_title(f'ADBE Volatility and Predictions\nFinal Volatility Prediction: {100*final_volatility_10:.2f}%', fontsize=12)
axs[1].set_xlabel('Days')
axs[1].set_ylabel('Volatility (%)')
axs[1].grid(axis='y')
axs[1].legend()

plt.tight_layout()
plt.show()
