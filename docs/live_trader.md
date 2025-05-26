# Live Trading Module (`live_trader.py`)

## 1. Overview

The `live_trader.py` script is designed to automate intraday option trading based on signals generated from the NIFTY 50 index. It operates in a continuous loop during market hours, fetches live market data, applies a configured trading strategy, places market orders for options, monitors active trades for stop-loss/profit-target conditions or strategy-based exits, and provides real-time notifications via Telegram.

## 2. Core Class: `LiveTrader`

The primary component is the `LiveTrader` class, which orchestrates all live trading operations.

### 2.1. Initialization (`__init__`)

Upon instantiation, `LiveTrader` performs the following:

*   **Configuration Loading**: Reads `trading_config.ini` to fetch all necessary parameters.
*   **Logging Setup (`_setup_logging`)**: Initializes logging to both a file (`live_trader.log`, overwritten on each run) and the console. Log messages include timestamps, log level, module name, line number, and the message.
*   **Component Initialization**:
    *   `OrderPlacement`: An instance of the `OrderPlacement` class from `myKiteLib.py` is created. Its `init_trading()` method is called to ensure the Kite API session is active, the access token is valid, and instrument data is loaded/refreshed.
    *   `DataPrep`: An instance of `DataPrep` from `trading_strategies.py` is created for calculating technical indicators.
*   **Strategy Loading (`_load_strategy_from_config`)**: 
    *   Reads the `active_strategy_config_section` from `[LIVE_TRADER_SETTINGS]` in the config file.
    *   Dynamically loads the strategy class (e.g., `DonchianBreakoutStrategy`) specified in the chosen strategy section.
    *   Initializes the strategy object with its specific parameters (e.g., `length`, `exit_option` for Donchian channels) also read from the config.
    *   Adjusts the global `NIFTY_DATA_FETCH_CANDLES` constant to ensure enough historical NIFTY data is fetched for the strategy's lookback period.
*   **State Variables**: Initializes internal state variables:
    *   `active_trade_details`: A dictionary to store all information about an ongoing trade.
    *   `is_trade_active`: Boolean, `True` if a trade is currently active.
    *   `trading_session_enabled`: Boolean, acts as a master switch for the main trading loop; can be set to `False` to gracefully shut down.
    *   `system_healthy`: Boolean, indicates if the Kite API is responsive.
    *   `trade_execution_allowed`: Boolean, determines if new trades can be initiated based on system health and trading hours.
*   **Key Parameters**: Loads and stores critical operational parameters from the config:
    *   NIFTY index token.
    *   Polling interval for the main loop.
    *   Trading start/end times and health check window times.
    *   Option type (e.g., 'CE'), trade units (number of lots).
    *   Profit target percentage, stop-loss percentage, and maximum trade holding period (minutes).

### 2.2. System Health & Session Management

*   **`_system_health_check(self)`**: 
    *   Called during the health check window (e.g., 09:15 - 09:20 AM) and if the system was unhealthy at the start of the trading window.
    *   Performs a simple Kite API call (fetching NIFTY historical data for the current day) to check API responsiveness.
    *   Sets `self.system_healthy` to `True` or `False`.
*   **`_manage_trading_session_state(self)`**: 
    *   Executed at the beginning of each main loop iteration.
    *   Checks the current time against configured health check and trading windows.
    *   Calls `_system_health_check()` if within the health check window and system isn't already marked healthy.
    *   Sets `self.trade_execution_allowed` based on whether the current time is within trading hours and `self.system_healthy` is `True`.
    *   Sets `self.trading_session_enabled = False` if the current time is past the `trading_end_time` to stop the main loop.

### 2.3. Main Trading Loop (`run_live_session`)

This is the heart of the `LiveTrader` and runs continuously while `self.trading_session_enabled` is `True`.

*   **Session State Management**: Calls `_manage_trading_session_state()`.
*   **NIFTY Data & Signals**:
    *   Calls `_get_latest_nifty_data()` to fetch recent 1-minute NIFTY candles.
    *   Passes this data to `self.data_prep.calculate_statistics()` to add indicators.
    *   The data with indicators is then passed to `self.strategy_obj.generate_signals()`.
    *   The signal (BUY: 1, SELL/EXIT: -1, HOLD: 0) from the most recent completed NIFTY candle is extracted along with the NIFTY price and timestamp at that signal.
    *   Sends Telegram notifications for new NIFTY BUY or active NIFTY EXIT signals.
*   **Trade Execution & Monitoring**:
    *   If `self.is_trade_active` is `True`: Calls `_monitor_active_trade()` with the latest NIFTY signal and candle data.
    *   Else if `self.trade_execution_allowed` is `True` and the latest NIFTY signal is BUY (1): Calls `_initiate_new_trade()`.
*   **Polling**: Pauses for `self.polling_interval_seconds` before the next iteration.
*   **Error Handling**: Includes `try-except` blocks for `KeyboardInterrupt` (for manual shutdown) and general exceptions to log errors and attempt a safe shutdown.

### 2.4. Data Fetching Methods

*   **`_get_latest_nifty_data(self)`**: 
    *   Fetches the last `NIFTY_DATA_FETCH_CANDLES` (a configurable constant, adjusted by strategy lookback) of 1-minute NIFTY index data using `self.order_manager.kite.historical_data()`.
    *   Returns a Pandas DataFrame, sorted by date, containing the most recent candles.
*   **`_get_last_option_candle(self, option_token: int)`**: 
    *   Fetches the most recent 1-minute candle data for the specified `option_token` (the one being traded) using `self.order_manager.kite.historical_data()`.
    *   Used by `_monitor_active_trade` to check for SL/TP conditions based on the option's high/low prices.
    *   Returns a Pandas Series representing the last candle.

### 2.5. Option Selection (`_find_live_option_token`)

*   Takes the NIFTY price at signal time and the signal datetime as input.
*   Constructs and executes a SQL query via `self.order_manager.con` (the database connection from `myKiteLib`) against the `instruments_zerodha` table.
*   The query selects the NIFTY option (CE or PE based on `self.option_type`) with the closest strike price to the current NIFTY price, for the current month's expiry (or nearest future expiry based on query logic).
*   Returns a tuple: `(tradingsymbol, instrument_token, lot_size)` or `None` if no suitable option is found.

### 2.6. Trade Initiation (`_initiate_new_trade`)

*   Called when a BUY signal on NIFTY is received, no trade is currently active, and `self.trade_execution_allowed` is `True`.
*   Calls `_find_live_option_token()` to select an option contract.
*   If an option is found:
    *   Calculates the actual trade quantity (`self.trade_quantity_actual = self.trade_units * option_lot_size`).
    *   **Entry Order Placement**: Attempts to place a MARKET BUY order for the option using `self.order_manager.place_market_order_live()`.
        *   **Retry Logic**: If the order placement fails, it retries up to `MAX_ENTRY_ORDER_RETRIES` (default 2) times with a `ENTRY_ORDER_RETRY_DELAY_SECONDS` (default 3) delay between attempts.
        *   Sends a Telegram message upon successful order placement or if all retry attempts fail.
    *   If an `entry_order_id` is successfully obtained:
        *   Sets `self.is_trade_active = True`.
        *   Populates `self.active_trade_details` with initial trade information: option symbol, token, lot size, actual quantity, Kite entry order ID, NIFTY signal details, current status (`PENDING_ENTRY_CONFIRMATION`), initiation time, and the calculated `max_hold_exit_time`.

### 2.7. Trade Monitoring (`_monitor_active_trade`)

This method manages the lifecycle of an active trade and has several states:

*   **If `status == 'PENDING_ENTRY_CONFIRMATION'`**: 
    *   Fetches the history of the entry order ID using `self.order_manager.get_order_history_live()` and associated trades using `self.order_manager.get_trades_for_order_live()`.
    *   If the order status is 'COMPLETE' and fill details (average price, filled quantity) are valid:
        *   Updates `self.active_trade_details` with the actual entry price and timestamp.
        *   Calculates and stores the `sl_price_calculated` and `target_price_calculated` based on the actual entry price and configured percentages.
        *   Changes status to `ACTIVE`.
        *   Sends a Telegram message confirming trade entry.
    *   If the order is 'REJECTED' or 'CANCELLED', it calls `_clear_active_trade_details()` to log the failure and reset.
*   **If `status == 'ACTIVE'`**: 
    *   Fetches the latest 1-minute candle for the traded option using `_get_last_option_candle()`.
    *   Checks for exit conditions in the following order of priority:
        1.  **Stop-Loss**: If `current_option_candle['low'] <= self.active_trade_details['sl_price_calculated']`.
        2.  **Target Price**: If `current_option_candle['high'] >= self.active_trade_details['target_price_calculated']`.
        3.  **Strategy Exit Signal**: If the latest NIFTY signal (`latest_nifty_signal`) is -1 (EXIT).
        4.  **Max Holding Period**: If `datetime.now() >= self.active_trade_details['max_hold_exit_time']`.
    *   If any exit condition is met:
        *   Logs the reason.
        *   **Exit Order Placement**: Attempts to place a MARKET SELL order for the option using `self.order_manager.place_market_order_live()`.
        *   If successful (an `exit_order_id` is obtained), updates `self.active_trade_details` with the exit order ID, trigger reason, and sets status to `PENDING_EXIT_CONFIRMATION`. Sends a Telegram message about the exit order placement.
        *   **Critical Exit Failure**: If placing the exit order fails, it logs a CRITICAL error, sends a Telegram alert, and **terminates the script using `sys.exit()`** to prevent further unintended behavior with an open position.
*   **If `status == 'PENDING_EXIT_CONFIRMATION'`**: 
    *   Fetches the history of the exit order ID.
    *   If the order status is 'COMPLETE' with valid fill details:
        *   Calls `_clear_active_trade_details()` with the actual exit price and reason.
    *   Handles cases where exit confirmation might be problematic (e.g., rejected exit market order, though unlikely).

### 2.8. Trade Clearing (`_clear_active_trade_details`)

*   Called when a trade is fully concluded (exit confirmed) or if it fails at an earlier stage (e.g., entry failed).
*   Performs comprehensive logging of all trade details: option symbol, NIFTY signal context, entry order ID/price/timestamp, exit order ID/price/timestamp, exit reason, quantity, calculated SL/TP targets.
*   Calculates and logs PNL (Profit and Loss) for the trade.
*   Sends a detailed trade summary via Telegram.
*   Resets `self.is_trade_active = False` and clears `self.active_trade_details`, `self.option_lot_size`, and `self.trade_quantity_actual` to prepare for a potential new trade.

## 3. Configuration (`trading_config.ini`)

The `LiveTrader` relies on settings from `trading_config.ini`, primarily from these sections:

*   **`[SIMULATOR_SETTINGS]`**: For `index_token`.
*   **`[LIVE_TRADER_SETTINGS]`**: For `polling_interval_seconds`, `trading_start_time`, `trading_end_time`, `health_check_start_time`, `health_check_end_time`, `active_strategy_config_section`, `product_type`.
*   **Strategy-Specific Section (e.g., `[STRATEGY_CONFIG_DonchianStandard]`)**: For `strategy_class_name`, indicator parameters (like `length`), `option_type`, `trade_units`, `profit_target_pct`, `stop_loss_pct`, `max_holding_period_minutes`.

## 4. Main Execution (`if __name__ == '__main__':`)

*   Locates the `trading_config.ini` file.
*   Instantiates the `LiveTrader` class.
*   Calls `trader.run_live_session()` to start the live trading loop.
*   Includes basic error handling for script startup issues.

## 5. Dependencies

Key external libraries:
*   `logging`, `time`, `datetime`, `configparser`, `os`, `pandas`, `numpy`, `sys`
*   Custom: `myKiteLib.OrderPlacement`, `trading_strategies.DataPrep`, `trading_strategies.DonchianBreakoutStrategy` (and other strategies).
*   From `myKiteLib`: `kiteconnect` (for Kite API interactions), `requests` (for Telegram). 