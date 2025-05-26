# Automated Trading System Implementation

## Project Objective

The primary goal is to develop an automated trading system that:
1.  Fetches real-time data from the Kite API.
2.  Analyzes data based on predefined technical strategies.
3.  Executes and manages trades.

## System Components

The system will be divided into the following main parts:

1.  **Data Preparation**: Involves acquiring, cleaning, and storing market data. This will be supported by a `DataPrep` class within a dedicated `trading_strategies.py` file, responsible for fetching and preparing data for strategy analysis.
2.  **Strategy Definition & Signal Generation (`trading_strategies.py`)**:
    *   A new file, `trading_strategies.py`, will house all strategy-related logic.
    *   It will include:
        *   The `DataPrep` class mentioned above.
        *   A `BaseStrategy` abstract class to define a common interface for all trading strategies (e.g., a `generate_signals(self, data)` method).
        *   Individual strategy classes (e.g., `StrategyReversalV1`) inheriting from `BaseStrategy`, each implementing its unique logic for generating trading signals.
3.  **Trading Loop & Orchestration**:
    *   A central orchestrator script will manage the overall workflow.
    *   **Trading Simulation (`TradingSimulator`)**:
        *   A dedicated `TradingSimulator` module will be developed.
        *   This simulator will instantiate `DataPrep` and strategy classes from `trading_strategies.py`.
        *   It will be responsible for running strategies against historical or real-time data, simulating trades, and evaluating performance.
        *   A `StrategyManager` (or similar component) within the `TradingSimulator` will manage the lifecycle of strategies, pass data to them, and collect generated signals.
        *   The output of the simulation (often a "signal file" or performance report) will be generated here.
    *   **Trade Execution & Monitoring**: Handles the actual placement of trades via the Kite API and monitors their performance.

The existing `myKiteLib.py` will serve as a foundational library for basic Kite API interactions and other common functionalities, utilized by `DataPrep` and other components.

## Immediate Next Steps

1.  **Daily Options Data Ingestion (Minute Level)**:
    *   Develop a daily loop to run every evening to fetch minute-level options data.
    *   **New Instruments**: Ensure new options instruments are identified and included.
    *   **Data Looping**: Implement a robust loop to fetch data for all relevant options.
    *   **One-Time Backfill**: Perform a one-time data backfill for the last two months.
    *   **Daily Delta**: Modify the loop to fetch data for the last two days on a daily basis.

2.  **Candle Data Resampling Function**:
    *   Create a function within `myKiteLib.py`.
    *   **Input**: Accept an instrument token or a list of tokens.
    *   **Process**: Fetch minute-level candle data from the database for the given token(s).
    *   **Output**: Convert the 1-minute data into 2-minute, 3-minute, 4-minute, 5-minute, and 10-minute intervals and return as a Pandas DataFrame.

# Implementation Details and Progress

This document tracks the implementation progress, technical details, and architectural decisions for the intraday options trading model.

## Recent Progress (as of last update)

Previously, the primary focus was on developing and refining the `TradingSimulator` and integrating it with a configuration-driven approach for strategies. The simulator was successfully run with the `DonchianBreakoutStrategy`.

**New Developments: Live Trading Module & Enhancements**

A significant new phase involved the development of a live trading module (`live_trader.py`) and supporting enhancements to the existing codebase.

1.  **`myKiteLib.py` Enhancements for Live Operations**:
    *   A new class `OrderPlacement` was introduced, inheriting from `system_initialization`.
    *   This class encapsulates live trading actions:
        *   `place_market_order_live()`: Places market orders.
        *   `get_order_history_live()`, `get_all_orders_live()`, `get_positions_live()`, `get_trades_for_order_live()`: Wrappers for respective Kite API calls to fetch order and position details.
    *   **Telegram Notifications**: 
        *   The `system_initialization` class now stores Telegram bot token and chat ID as instance variables.
        *   A `send_telegram_message(self, message: str)` method was added to `OrderPlacement` to send real-time updates.

2.  **Live Trading Module (`live_trader.py`) Implementation**:
    *   A new script `live_trader.py` containing the `LiveTrader` class was developed to automate live intraday option trading.
    *   **Core Functionality**:
        *   Operates in a polling loop during configurable market hours.
        *   Performs a system health check before and during trading hours by verifying Kite API responsiveness.
        *   Dynamically loads and applies a trading strategy (e.g., `DonchianBreakoutStrategy`) from `trading_strategies.py` based on `trading_config.ini`.
        *   Fetches live 1-minute NIFTY 50 index data, calculates indicators, and generates BUY/SELL signals.
        *   Selects the closest appropriate NIFTY option (CE, based on config) for the current month using a database query for instrument details (including `lot_size`).
        *   **Trade Execution**: Places MARKET orders for option entry and exit.
            *   **Entry Retry**: Implements a retry mechanism (default 2 retries) for entry order placement if initial attempts fail.
        *   **Trade Management (Client-Side)**:
            *   Monitors active trades by fetching the latest 1-minute candle of the traded option.
            *   Checks for Stop-Loss (SL) and Target Price (TP) conditions based on the option's low/high prices against calculated SL/TP levels.
            *   Monitors for strategy-generated EXIT signals on the NIFTY index.
            *   Enforces a maximum trade holding period.
        *   **Critical Error Handling**: 
            *   If placing an *exit* order fails, the script logs a critical error, sends a Telegram alert, and terminates immediately (`sys.exit()`) to prevent unmanaged open positions.
        *   **Logging**: Comprehensive logging to both console and a file (`live_trader.log`, overwritten on each run).
        *   **Telegram Notifications**: Sends real-time alerts for:
            *   NIFTY BUY/EXIT signals.
            *   Entry order placement attempts (success/all retries failed).
            *   Entry order confirmation.
            *   Exit order placement.
            *   Critical exit order failures.
            *   Detailed trade closure summaries including PNL.
    *   **Configuration Driven**: Highly configurable via `trading_config.ini`, including trading hours, polling intervals, active strategy, product type, and strategy parameters.
    *   **Detailed Documentation**: For an in-depth understanding of the `LiveTrader` module, refer to [`docs/live_trader.md`](./live_trader.md).

3.  **Concurrency Logic in `TradingSimulator` (Previous Update)**:
    *   The `TradingSimulator` in `trading_simulator.py` was previously refactored to correctly handle non-concurrent trades by tracking the actual simulated exit time of an active trade when `allow_concurrent_trades` is `False`.

## Technical Details

### Core Components:

*   **`trading_config.ini`**: Central configuration file.
    *   `[SIMULATOR_SETTINGS]`: Global settings for the simulator.
    *   `[DATA_PREP_DEFAULTS]`: Default parameters for `DataPrep` if not overridden by strategy needs.
    *   `[STRATEGY_CONFIG_*]`: Sections for each strategy, defining:
        *   `strategy_class_name`: The Python class for the strategy.
        *   Indicator-specific parameters (e.g., `length` for Donchian).
        *   Trading parameters (option type, interval, units, PNL targets, holding period).

*   **`trading_simulator.py`**:
    *   **`TradingSimulator` Class**:
        *   `__init__`: Initializes with index token, instantiated strategy object, trade dates, option type, interval, specific trade parameters (profit target, stop loss, etc.), and initial capital. All these are derived from `trading_config.ini`.
        *   `_find_closest_CE_option` / `_find_closest_PE_option`: Selects the nearest strike option (CE or PE) based on the NIFTY price at the signal time using SQL queries against the `instruments_zerodha` table. Queries filter for options expiring in the current month of the signal.
        *   `_simulate_single_trade_on_option`:
            *   Takes option OHLCV data, NIFTY signal time, NIFTY price, and a series of NIFTY exit signals for the trade window.
            *   Option Entry: At the `open` price of the option candle at or immediately following the NIFTY BUY signal.
            *   Exit Conditions (checked per minute, in order of priority):
                1.  **NIFTY Strategy Exit Signal**: If the underlying NIFTY strategy (e.g., Donchian) generates an explicit exit signal (`-1`) during the holding period.
                2.  **Profit Target**: If `option_high >= entry_price * (1 + profit_target_pct)`. Exit at target price.
                3.  **Stop Loss**: If `option_low <= entry_price * (1 - stop_loss_pct)`. Exit at stop-loss price.
                4.  **Max Holding Period**: If `max_holding_period_minutes` is reached. Exit at `close` of the last candle.
                5.  **End of Option Data**: If option data for the selected token runs out. Exit at `close` of the last available candle.
            *   Logs detailed information for each simulated trade.
        *   `run_simulation`:
            1.  Fetches and prepares NIFTY index data using `DataPrep.fetch_and_prepare_data` for the specified `trade_interval`.
            2.  Calculates technical indicators on the NIFTY data using `DataPrep.calculate_statistics` (leveraging strategy-specific parameters like `length` if provided by the strategy object).
            3.  Generates BUY/SELL/HOLD signals on NIFTY data using the `generate_signals` method of the instantiated strategy object.
            4.  Iterates through BUY signals on NIFTY:
                *   Finds the appropriate option token (CE/PE based on config).
                *   Fetches option OHLCV data using `DataPrep.fetch_and_prepare_data` for the period from the signal date to the simulation end date.
                *   Calls `_simulate_single_trade_on_option` to simulate the trade using the fetched option data and NIFTY exit signals.
            5.  Collects all executed trades into a DataFrame.
        *   `calculate_performance_metrics`: Computes metrics like win rate, PNL, profit factor, max drawdown.
        *   `save_results`: Saves the detailed trades log (CSV) and performance summary (TXT) to the `cursor_logs/` directory.
    *   **`if __name__ == '__main__':` Block**:
        *   Loads global configuration from `trading_config.ini`.
        *   Selects a strategy configuration section (e.g., `STRATEGY_CONFIG_DonchianStandard`).
        *   Extracts simulator settings and strategy-specific parameters.
        *   Dynamically instantiates the chosen strategy class (e.g., `DonchianBreakoutStrategy`) with its indicator parameters.
        *   Instantiates `TradingSimulator` with all necessary configured parameters.
        *   Calls `run_simulation`, then `calculate_performance_metrics` and `save_results`.

*   **`trading_strategies.py`**:
    *   **`DataPrep` Class**:
        *   `fetch_and_prepare_data`:
            *   Fetches raw 1-minute OHLCV data from the MySQL database (`myKiteLib.get_historical_data_from_db_for_token_and_interval_new`).
            *   Handles column renaming, type conversion, sorting, and dropping duplicates.
            *   If `interval` is > 1 minute, calls `convert_minute_data_interval` to resample.
        *   `convert_minute_data_interval`: Resamples 1-minute data to '3minute', '5minute', etc.
        *   `calculate_statistics`: Calculates technical indicators (e.g., Donchian Channels via `_add_donchian_channels`). Uses strategy-specific parameters if available (e.g., `donchian_length`), otherwise falls back to defaults from `[DATA_PREP_DEFAULTS]` in `trading_config.ini`.
    *   **`DonchianBreakoutStrategy` Class**:
        *   `__init__`: Takes `length` and `exit_option` as parameters (from config).
        *   `generate_signals`:
            *   Requires DataFrame with 'high', 'low', 'close', 'Donchian_Upper', 'Donchian_Lower', 'Donchian_Mid'.
            *   BUY signal (1): If `close` crosses above `Donchian_Upper`.
            *   SELL/EXIT signal (-1):
                *   If `exit_option == 1`: If `close` crosses below `Donchian_Lower`.
                *   If `exit_option == 2`: If `close` crosses below `Donchian_Mid`.
            *   HOLD signal (0): Otherwise.

*   **`myKiteLib.py`**:
    *   Provides database connection (`mysqlDB`) and data fetching utilities (`kiteAPIs.get_historical_data_from_db_for_token_and_interval_new`).
    *   The `UserWarning`s and shutdown errors observed might originate from interactions within this library or how it's used by `DataPrep` and `TradingSimulator`.

## Next Steps / Areas for Improvement

*   **Live Trader Robustness**: Further testing and hardening of the `LiveTrader` module with real market data and conditions.
    *   Consider more sophisticated error handling for non-critical API issues (e.g., temporary network glitches for non-order-placement calls).
    *   Implement state persistence for `active_trade_details` in `LiveTrader` to allow recovery from script restarts.
*   Investigate and resolve SQLAlchemy `UserWarning`s.
*   Investigate and resolve MySQL connection shutdown `TypeError`s.
*   Expand strategy library with more complex indicators and logic.
*   Implement more sophisticated risk management and position sizing (beyond fixed units/lots).
*   Enhance reporting and visualization of simulation and live trading results.
*   Consider parameter optimization techniques for strategies.

## Live Trader Pre-Run Checklist

Before running the `live_trader.py` script, ensure the following checks are performed:

1.  **`trading_config.ini` Verification**:
    *   **`[SIMULATOR_SETTINGS]` -> `index_token`**: Confirm it's set to the NIFTY 50 index token (e.g., 256265).
    *   **`[LIVE_TRADER_SETTINGS]`**: 
        *   `polling_interval_seconds`: Appropriate for your needs (e.g., 30-60).
        *   `trading_start_time`, `trading_end_time`: Correct market hours (e.g., 09:20:00 - 15:00:00).
        *   `health_check_start_time`, `health_check_end_time`: Correct pre-market check window (e.g., 09:15:00 - 09:20:00).
        *   `active_strategy_config_section`: Points to the correct `[STRATEGY_CONFIG_*]` section you intend to use.
        *   `product_type`: Set correctly (e.g., `MIS` for intraday).
    *   **Strategy Configuration Section (e.g., `[STRATEGY_CONFIG_DonchianStandard]`)**: Referenced by `active_strategy_config_section`.
        *   `strategy_class_name`: Correct strategy class (e.g., `DonchianBreakoutStrategy`).
        *   Indicator parameters (e.g., `length`, `exit_option` for Donchian) are correctly set.
        *   `option_type`: Typically `CE` for the current setup focusing on long calls.
        *   `trade_units`: Number of lots for trading.
        *   `profit_target_pct`, `stop_loss_pct`: Correct percentage values (e.g., 0.01 for 1%).

2.  **`myKiteLib.py` & `security.txt`**: 
    *   Ensure `security.txt` (or the path configured in `myKiteLib.py`) contains valid and up-to-date Kite API credentials (`api_key`, `api_secret`, `userID`, `pwd`, `totp_key`).
    *   The `AccessToken` in `security.txt` should be recent, or confirm the auto-renewal logic within `system_initialization.init_trading()` is functioning reliably.
    *   Telegram tokens (`telegramToken_global`, `telegramChatId_global` in `myKiteLib.py` or as configured) are correct for notifications.

3.  **Database Connectivity**:
    *   The MySQL server must be running and accessible.
    *   Credentials in `security.txt` for database connection (`username`, `password`, `hostname`, `port`, `database_name`) must be correct.
    *   The `instruments_zerodha` table should be populated and up-to-date for option contract lookups.

4.  **Python Environment & Dependencies**:
    *   The correct virtual environment (e.g., `/opt/anaconda3/envs/KiteConnect`) must be activated.
    *   All required Python packages (`pandas`, `numpy`, `kiteconnect`, `ta`, `mysql-connector-python`, `SQLAlchemy`, `PyMySQL`, `selenium`, `pyotp`, `requests`) must be installed in the environment.

5.  **System Time Synchronization**:
    *   Ensure the machine running the script has its system time accurately synchronized with an NTP server to avoid issues with trading window checks and candle data alignment.

6.  **Code Sanity Check (If any recent manual changes)**:
    *   Briefly review `live_trader.py`, `myKiteLib.py`, and `trading_strategies.py` for any obvious errors if manual edits were made outside of automated generation.

7.  **Monitoring Plan**:
    *   Be prepared to closely monitor the `live_trader.log` file during the session.
    *   Keep an eye on Telegram notifications for real-time updates.
    *   Have the Kite trading platform (web or mobile) accessible for manual oversight or intervention if necessary.

8.  **Capital and Risk**:
    *   Ensure sufficient capital is available in the trading account.
    *   Be aware of the risk settings (`trade_units`, `stop_loss_pct`) and trade with amounts you are comfortable with, especially during initial live runs.

## New Strategy Development: Fibonacci Retracement Trend Continuation

**Objective**: Identify pullbacks to Fibonacci levels within an established trend on the NIFTY50 index (identified by Higher Highs and Higher Lows) and enter call option positions anticipating a resumption of the trend. This strategy focuses on short-term (5-10 minute) reversals/continuations.

**Core Logic**:

1.  **Swing Point Identification (Zig Zag Pattern)**:
    *   Develop a robust algorithm to identify significant swing highs and swing lows in the NIFTY50 index price action (e.g., on 1-minute or 5-minute data).
    *   Parameters for swing detection (e.g., number of bars, percentage/point deviation) will be configurable to capture swings that define a 10-minute to 60-minute zig-zag pattern.

2.  **Trend Identification (Market Structure)**:
    *   **Uptrend**: Confirmed by a sequence of Higher Highs (HH) and Higher Lows (HL) formed by the identified swing points.
    *   **Downtrend Reversal for Calls**:
        *   Identify an established downtrend (Lower Highs - LH, Lower Lows - LL).
        *   Look for a break in this pattern: a Low (LL), followed by a Higher Low (HL), and then a Higher High (HH). This signals a potential shift to an uptrend.

3.  **Fibonacci Retracement Application**:
    *   **In an Uptrend**: Once a new HH is formed after an HL, draw Fibonacci retracement levels from the most recent confirmed HL to the most recent confirmed HH.
    *   **Post Downtrend Reversal**: After the LL -> HL -> HH sequence confirms a potential new uptrend, draw Fibonacci from this new HL to the new HH.

4.  **Entry Signal (Buying Call Options)**:
    *   **Condition 1 (Trend)**: NIFTY50 index is in a confirmed uptrend (HH/HL structure) or has just signaled a downtrend reversal (LL->HL->HH).
    *   **Condition 2 (Pullback)**: Price retraces from the recent swing high (HH) and approaches a key Fibonacci support level (e.g., 38.2%, 50.0%, 61.8%).
    *   **Condition 3 (Confirmation)**: Bullish confirmation at the Fibonacci level (e.g., bullish candlestick pattern, RSI divergence/move from oversold, volume surge).
    *   **Action**: Buy a NIFTY50 Call option.

5.  **Option Selection**:
    *   **Strike Price**: At-the-money (ATM) or slightly out-of-the-money (OTM) call options.
    *   **Expiry**: Weekly expiry, with sufficient time to avoid rapid decay (e.g., 2-3 days minimum, unless scalping).

6.  **Stop Loss**:
    *   **Index-Based**: Below the Fibonacci level that provided support, or below the low of the confirmation candle/recent swing low.
    *   **Option-Based**: Percentage of option premium or derived from index stop-loss.

7.  **Target Price**:
    *   Previous swing high.
    *   Fibonacci extension levels (if previous swing high is broken).
    *   Fixed Risk-Reward ratio.

**Implementation Phases for Fibonacci Strategy**:

*   **Phase F1: Core Price Action Analysis Module Enhancement**
    *   Task F1.1: Design and implement the configurable Swing Point Detection algorithm. This is critical for identifying HH/HL/LH/LL.
    *   Task F1.2: Ensure Fibonacci calculation is robust and can be applied to dynamic swing points.
    *   Task F1.3: Integrate optional confirmation indicators (e.g., RSI).

*   **Phase F2: Trend & Pattern Identification Logic**
    *   Task F2.1: Implement logic to identify uptrends (HH/HL sequences) and downtrends (LH/LL sequences) based on swing points.
    *   Task F2.2: Implement logic to detect the specific "downtrend reversal" pattern (LL -> HL -> HH).

*   **Phase F3: Signal Generation & Trade Logic**
    *   Task F3.1: Develop the signal generation logic combining trend, Fibonacci levels, and confirmation.
    *   Task F3.2: Define option selection rules within the strategy.
    *   Task F3.3: Define SL/TP logic based on swing structure and Fibonacci levels.

*   **Phase F4: Backtesting & Refinement**
    *   Task F4.1: Adapt/use the existing backtesting environment for this new strategy.
    *   Task F4.2: Thoroughly backtest and analyze performance.
    *   Task F4.3: Iteratively refine swing detection parameters, Fibonacci levels, and confirmation rules.

*   **Phase F5: Documentation**
    *   Task F5.1: Create `docs/strategy_documentation/fibonacci_swing_trend.md` for in-depth details.
    *   Task F5.2: Keep this section of `implementation.md` updated with progress. 