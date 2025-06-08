# Trading Bot Implementation Details

This document provides a detailed overview of the trading bot's architecture, file structure, and the purpose of each key component.

## Architecture & Control Flow

The diagram below illustrates the application's control flow, starting from initialization and moving into the main trading loop. It shows the distinct paths taken when the bot is searching for a new trade versus when it is actively managing an open position.

```mermaid
graph TD
    subgraph "Phase 1: Initialization"
        A[Start main_live.py] --> B(Instantiate LiveTrader);
        B --> B1(Load Config);
        B --> B2(Init OrderManager);
        B --> B3(Init DataHandler);
        B --> B4(Init ZigZagHarmonicStrategy);
        B --> B5(Init SessionManager);
    end

    subgraph "Phase 2: LiveTrader.run_session() Main Loop"
        C(Start Loop) --> D{Is Market Open?};
        D -- No --> E(Wait for Next Interval...);
        E --> C;
        D -- Yes --> F(data_handler.fetch_latest_data);
        F --> G(Get Current LTP from Data);
        G --> H{Is a Trade Active?};
    end
    
    A --> C;

    subgraph "Scenario B: Active Trade Exists"
        H -- Yes --> O(position_manager.check_for_exit);
        O --> P{Target or Stop-Loss Hit?};
        P -- No --> E;
        P -- Yes --> Q(order_manager.place_exit_order);
        Q -- Order Fails --> E;
        Q -- Order Succeeds --> R(position_manager.close_trade);
        R --> S(Calculate & Log PnL);
        S --> E;
    end
    
    subgraph "Scenario A: No Active Trade - Signal Generation"
        H -- No --> S_Start(strategy.check_for_signal(data, ltp));
        
        subgraph "Inside check_for_signal (Stateful Logic)"
            S_Start --> S_Resample{Use Alt. Timeframe?};
            S_Resample -- Yes --> S_Resample_Action(Resample data to N-min);
            S_Resample -- No --> S_CalcPivots;
            S_Resample_Action --> S_CalcPivots(indicators.calculate_zigzag_pivots);

            S_CalcPivots --> S_CheckPivotCount{"Sufficient pivots? (>=5)"};
            S_CheckPivotCount -- No --> S_ClearAndFail(Clear Pending Signal & Return None);

            S_CheckPivotCount -- Yes --> S_CheckPending{Pending Signal Exists?};

            subgraph "Path 1: Pending Signal Exists"
                S_CheckPending -- Yes --> S_ValidatePending{Is D-pivot still the same?};
                S_ValidatePending -- No --> S_ClearAndEval(Clear Pending & Evaluate New);
                S_ValidatePending -- Yes --> S_CheckWindow{LTP in Entry Window?};
                S_CheckWindow -- No --> S_Fail(Return None);
                S_CheckWindow -- Yes --> S_ExecSignal(Return Executable Signal);
            end

            subgraph "Path 2: No Pending Signal"
                S_CheckPending -- No --> S_EvalNewPattern;
                S_ClearAndEval --> S_EvalNewPattern{Is D-pivot new?};
                S_EvalNewPattern -- No --> S_Fail;
                S_EvalNewPattern -- Yes --> S_FindPattern{Harmonic Pattern Found?};
                S_FindPattern -- No --> S_Fail;
                S_FindPattern -- Yes --> S_SetPending(Create & Store Pending Signal);
                S_SetPending --> S_Fail;
            end
        end
        
        subgraph "Processing the Signal in LiveTrader"
            S_ClearAndFail --> J{Executable Signal Returned?};
            S_Fail --> J;
            S_ExecSignal --> J;
            J -- No --> E;
            J -- Yes --> L(order_manager.place_entry_order);
            L -- Order Fails --> E;
            L -- Order Succeeds --> M(position_manager.start_new_trade);
            M --> E;
        end
    end
```

## 1. Project Structure

The project is organized into several directories to separate concerns, making the codebase modular and maintainable.

```
.
├── config/                  # Configuration files
├── logs/                    # Application and strategy debug logs
├── scripts/                 # Standalone utility scripts (e.g., data backfilling)
├── strategy/                # Core trading strategy logic
├── trader/                  # Core trading execution components
├── .gitignore
├── implementation.md        # This file
├── live_trader_main.log     # Main log file for the application
├── main_live.py             # Main entry point for the application
├── security.txt             # Sensitive credentials (API keys, etc.)
└── zigzag.pine              # Original PineScript strategy file
```

## 2. Core Components

### `main_live.py`

- **Objective:** The main entry point for starting the trading bot.
- **Functionality:**
    - Parses command-line arguments, primarily to enable **replay mode** (`--replay "YYYY-MM-DD HH:MM:SS"`).
    - Sets up application-wide logging.
    - Loads the main configuration from `config/trading_config.ini`.
    - Instantiates and runs the `LiveTrader` class, which contains the main trading loop.
    - Handles top-level exceptions to ensure graceful shutdown.

### `config/`

- **`trading_config.ini`**: The primary configuration file for the bot. It contains settings for the trading session, strategy parameters (like TP/SL rates), alternate timeframes, and data lookback periods.

### `security.txt`

- **Objective:** To store sensitive information securely, keeping it separate from the main codebase.
- **Contents:** API keys, user credentials, database connection details, and the KiteConnect `AccessToken`. This file should be included in `.gitignore` and never committed to version control.

## 3. `trader/` Directory

This directory contains the core components responsible for the mechanics of trading, such as placing orders, managing the trading session, and handling data.

### `live_trader.py`

- **`LiveTrader` Class:**
    - **Objective:** Orchestrates the entire trading session. It houses the main event loop.
    - **Functionality:**
        - Initializes all major components: `OrderManager`, `DataHandler`, `ZigZagHarmonicStrategy`, and `SessionManager`.
        - The `run_session()` method contains the main `while` loop that drives the bot.
        - In each loop iteration, it checks if the market is open, fetches the latest market data, checks for trade signals, and manages the active trade's state (entry, exit).
        - Manages the internal clock for **replay mode**.

### `order_manager.py`

- **`OrderManager` Class:**
    - **Objective:** To handle all aspects of order execution and trade state management.
    - **Functionality:**
        - Tracks the currently active trade.
        - `place_new_trade()`: Places a new entry order via the Kite API and creates an `ActiveTrade` object to track its state.
        - `check_and_update_open_trade()`: Monitors the active trade's P&L against its target and stop-loss levels.
        - `close_trade()`: Places an exit order when a target or stop-loss is hit.
        - Manages the `trade_log.csv`.

### `session_manager.py`

- **`SessionManager` Class:**
    - **Objective:** To manage the trading session times.
    - **Functionality:**
        - `is_market_open()`: Checks if the current time is within the allowed trading hours defined in the config, ensuring the bot is only active when it should be.

### `data_handler.py`

- **`DataHandler` Class:**
    - **Objective:** To act as the single source for all market data.
    - **Functionality:**
        - `fetch_latest_data()`: Fetches the most recent 1-minute candle data required for the strategy to make a decision. In replay mode, it fetches data up to the specific replay timestamp.
        - `fetch_historical_data()`: Fetches larger blocks of historical data, primarily for backtesting purposes.
        - Includes data cleaning and validation logic.

### `myKiteLib.py`

- **Objective:** A library for all interactions with the KiteConnect API and the database.
- **Classes & Functions:**
    - **`system_initialization`**: Handles the initial login process, including generating a new `AccessToken` if the existing one is expired. Reads credentials from `security.txt`.
    - **`OrderPlacement`**: A wrapper around KiteConnect API calls for placing and managing orders (`place_market_order_live`, `get_positions_live`, etc.). Also handles sending Telegram notifications.
    - **`kiteAPIs`**: Provides methods for fetching data from the Kite API or the local database cache.
    - **`convert_minute_data_interval()`**: A standalone utility function to resample 1-minute data into any higher timeframe (e.g., 10-minute).

## 4. `strategy/` Directory

This directory contains all the logic specific to the ZigZag Harmonic trading strategy.

### `zigzag_harmonic.py`

- **`ZigZagHarmonicStrategy` Class:**
    - **Objective:** To implement the core signal generation logic.
    - **`check_for_signal()` method:**
        - Takes the latest market data from the `DataHandler`.
        - Determines if an alternate timeframe should be used and passes the correct resampling interval to the indicator function.
        - Calls `calculate_zigzag_pivots()` to identify significant highs and lows.
        - Uses the last 5 pivots to calculate Fibonacci ratios.
        - Iterates through all harmonic patterns defined in `patterns.py` to check for a match.
        - If a valid pattern is found, it constructs and returns a trade signal dictionary.

### `indicators.py`

- **`calculate_zigzag_pivots()` Function:**
    - **Objective:** Implements the ZigZag indicator logic based on the `zigzag.pine` script.
    - If a resampling interval is provided, it first uses `convert_minute_data_interval` to transform the data.
    - It then analyzes the OHLC data to identify pivot points (significant highs and lows) and returns them as a list.

### `patterns.py`

- **Objective:** Defines the mathematical conditions for each harmonic pattern.
- **Functionality:**
    - Contains a separate function for each pattern (e.g., `is_bat`, `is_gartley`). Each function takes the calculated Fibonacci ratios and returns `True` if the pattern's rules are met.
    - `ALL_PATTERNS`: A list that aggregates all pattern-checking functions, allowing the strategy to iterate through them easily.

### `strategy_logger.py`

- **`StrategyLogger` Class:**
    - **Objective:** Provides detailed, structured logging for debugging the strategy's decisions.
    - **Functionality:**
        - Writes logs to unique CSV files in the `logs/strategy_debug/` directory for each run.
        - `log_pivots()`: Logs the five pivots being used for a pattern check.
        - `log_pattern_check()`: Logs the ratios for every pattern check and whether it was a match.
        - `log_signal()`: Logs the full details of a generated trade signal.

## 5. Other Files & Directories

- **`scripts/data_backfill.py`**: A utility script to download historical data from the Kite API and store it in the local database, ensuring the bot has a rich data cache for backtesting and replay.
- **`zigzag.pine`**: The original PineScript file from TradingView. It serves as the authoritative source for the strategy's logic and rules.
- **`logs/`**: Contains the main application log and the strategy debugging CSVs.

--- 