import re
from ib_insync import IB, Stock, Option, MarketOrder

# Initialize IB connection
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=12)

# User's budget per position
BUDGET_PER_POSITION = 2000

# Helper function to calculate the number of contracts based on the option price
def calculate_contracts(option_price):
    if option_price is None or option_price <= 0:
        return 1  # Default to 1 contract if the price is not specified or invalid
    return int(BUDGET_PER_POSITION / (option_price * 100))

# List of orders based on user specifications
orders_to_place = [
    {
        "ticker": "PFE",
        "position_type": "Long",
        "expiration": "2024-11-19",
        "strike_price": 25,
        "option_type": "C",
        "option_price": 0.45
    },
    {
        "ticker": "TMO",
        "position_type": "Long",
        "expiration": "2024-11-29",
        "strike_price": 520,
        "option_type": "C",
        "option_price": None  # Price not specified
    },
    {
        "ticker": "PEP",
        "position_type": "Long",
        "expiration": "2024-12-13",
        "strike_price": 165,
        "option_type": "C",
        "option_price": None  # Smaller position, price not specified
    }
]

# Function to place the orders
def place_orders(orders):
    for order in orders:
        try:
            ticker = order['ticker']
            position_type = order['position_type']
            strike_price = order['strike_price']
            option_type = order['option_type']
            expiration = order['expiration']
            option_price = order.get('option_price')

            # Calculate the number of contracts
            contracts = calculate_contracts(option_price)

            # Define the option contract
            contract = Option(ticker, expiration, strike_price, option_type, 'SMART')

            # Check connection before qualifying the contract
            if not ib.isConnected():
                print(f"Not connected to Interactive Brokers. Skipping order for {ticker}.")
                continue

            # Qualify the contract with a timeout
            try:
                ib.qualifyContracts(contract, timeout=10)
            except Exception as e:
                print(f"Error qualifying contract for {ticker}: {e}")
                continue

            # Define the order type (buy for long)
            action = 'BUY' if position_type.lower() == 'long' else 'SELL'
            market_order = MarketOrder(action, contracts)

            # Place the order
            trade = ib.placeOrder(contract, market_order)
            print(f"Placed order for {ticker}: {contracts} contracts of {strike_price} {option_type} expiring {expiration}")

        except Exception as e:
            print(f"Error placing order for {ticker}: {e}")

# Place the specified orders
place_orders(orders_to_place)

# Disconnect from IB
ib.disconnect()
