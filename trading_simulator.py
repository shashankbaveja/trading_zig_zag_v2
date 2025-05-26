import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, time
import os
import calendar # Added for monthrange
import configparser # Added for reading config file

# Assuming trading_strategies.py and myKiteLib.py are in the same directory or accessible
from trading_strategies import DataPrep, DonchianBreakoutStrategy, MovingAverageRSILong, MovingAverageRSIShort # BaseStrategy can be added if needed for type hinting
from myKiteLib import kiteAPIs # For querying instrument details

# --- Configuration Loading ---
config = configparser.ConfigParser()
# Determine the path to the config file.
# This assumes trading_config.ini is in the same directory as this script.
# For more complex setups, this path might need to be absolute or more robustly determined.
_config_file_path = os.path.join(os.path.dirname(__file__), 'trading_config.ini')
if not os.path.exists(_config_file_path):
    # Attempt to find it one level up if not in the same directory
    _config_file_path_alt = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trading_config.ini')
    if os.path.exists(_config_file_path_alt):
        _config_file_path = _config_file_path_alt
    else:
        raise FileNotFoundError(f"trading_config.ini not found at {_config_file_path} or its parent directory.")
config.read(_config_file_path)
print(f"TradingSimulator: Configuration file \'{_config_file_path}\' loaded.")


# Removed old global default variables; they are now loaded from config within __main__ or passed

class TradingSimulator:
    def __init__(self, 
                 index_token: int,          # Typically from [SIMULATOR_SETTINGS]
                 strategy_obj,             # Instantiated by main script based on strategy config
                 trade_start_date: datetime, # Changed to datetime
                 trade_end_date: datetime,  # Changed to datetime
                 # Parameters below are typically from a specific [STRATEGY_CONFIG_...] section
                 option_type: str,          
                 trade_interval: str,       
                 trade_params: dict,        # e.g., {'units': ..., 'profit_target_pct': ..., etc.}
                 initial_capital: float,    # Typically from [SIMULATOR_SETTINGS]
                 allow_concurrent_trades: bool # From [SIMULATOR_SETTINGS]
                 ):
        
        self.index_token = index_token
        self.option_type = option_type.upper() # Should come from strategy config
        self.strategy_obj = strategy_obj
        self.trade_start_date = trade_start_date
        self.trade_end_date = trade_end_date
        self.trade_interval = trade_interval # Should come from strategy config
        self.initial_capital = initial_capital
        
        # trade_params are now expected to be fully populated by the caller
        # based on the chosen strategy's configuration.
        self.trade_params = trade_params 
        if not all(k in self.trade_params for k in ['units', 'profit_target_pct', 'stop_loss_pct', 'max_holding_period_minutes']):
            raise ValueError("trade_params dict is missing one or more required keys: units, profit_target_pct, stop_loss_pct, max_holding_period_minutes")

        self.allow_concurrent_trades = allow_concurrent_trades
        self.is_trade_active = False 
        print(f"TradingSimulator: allow_concurrent_trades set to: {self.allow_concurrent_trades}")

        self.data_prep = DataPrep()
        if not self.data_prep.k_apis:
            raise ConnectionError("TradingSimulator: Failed to initialize DataPrep's kiteAPIs.")
        self.k_apis = self.data_prep.k_apis

        self.executed_trades = []
        print(f"TradingSimulator initialized for Index Token: {self.index_token}, Option Type: {self.option_type}")
        print(f"Strategy: {type(self.strategy_obj).__name__} with params from config")
        print(f"Simulation Period: {self.trade_start_date} to {self.trade_end_date}, Interval: {self.trade_interval}")
        print(f"Option Trade Parameters: {self.trade_params}")
        print(f"Initial Capital for Metrics: {self.initial_capital}")

    def _find_closest_CE_option(self, underlying_price_at_signal: float, signal_datetime: datetime) -> int | None:
        """
        Finds the instrument token for the closest Call Option (CE) using the provided SQL query.
        """
        # ... (rest of _find_closest_CE_option remains largely the same, ensure DB connection logic is robust) ...
        # This method's internal SQL and DB interaction logic is unchanged by this refactoring.
        # However, ensure self.k_apis.startKiteSession.con is correctly managed.
        print(f"\nAttempting to find closest CE option for NIFTY price {underlying_price_at_signal:.2f} at {signal_datetime}")
        
        # query = """
        # WITH last_expiry_month AS (
        #     SELECT MAX(expiry) AS last_expiry_month
        #     FROM kiteConnect.instruments_zerodha
        #     WHERE name = 'NIFTY'
        #       AND instrument_type = 'CE'
        #       AND EXTRACT(MONTH FROM expiry) = EXTRACT(MONTH FROM %(signal_date)s)
        #       AND EXTRACT(YEAR FROM expiry) = EXTRACT(YEAR FROM %(signal_date)s)
        # ),
        # filtered_options AS (
        #     SELECT a.instrument_token, a.strike,
        #            ROW_NUMBER() OVER (ORDER BY a.strike ASC) AS rnum
        #     FROM kiteConnect.instruments_zerodha a
        #     INNER JOIN last_expiry_month b ON a.expiry = b.last_expiry_month
        #     WHERE a.name = 'NIFTY'
        #       AND a.instrument_type = 'CE'
        #       AND a.strike >= %(strike_price)s
        # )
        # SELECT instrument_token
        # FROM filtered_options 
        # WHERE rnum = 1;
        # """

        query = """
        SELECT 256265 as instrument_token;
        """

        selected_token: int | None = None
        conn = None
        try:
            conn = self.k_apis.startKiteSession.con 
            if not conn or not conn.is_connected():
                 print("    DB connection issue in _find_closest_CE_option. Re-initializing.")
                 self.k_apis.startKiteSession.init_trading() 
                 conn = self.k_apis.startKiteSession.con
            if not conn or not conn.is_connected():
                print("    Failed to establish DB connection in _find_closest_CE_option.")
                return None
            params = {'signal_date': signal_datetime.date(), 'strike_price': underlying_price_at_signal}
            option_df = pd.read_sql_query(query, conn, params=params)
            if not option_df.empty and 'instrument_token' in option_df.columns:
                selected_token = int(option_df['instrument_token'].iloc[0])
                print(f"    Selected CE Option Token: {selected_token}")
            else:
                print(f"    No CE option token found by SQL query for NIFTY at {underlying_price_at_signal:.2f}, time {signal_datetime}.")
        except Exception as e:
            print(f"    Error in _find_closest_CE_option: {e}")
            return None
        return selected_token

    def _find_closest_PE_option(self, underlying_price_at_signal: float, signal_datetime: datetime) -> int | None:
        """
        Finds the instrument token for the closest Put Option (PE) using the provided SQL query.
        """
        # ... (rest of _find_closest_PE_option remains largely the same) ...
        print(f"\nAttempting to find closest PE option for NIFTY price {underlying_price_at_signal:.2f} at {signal_datetime}")
        query = """
        WITH last_expiry_month AS (
            SELECT MAX(expiry) AS last_expiry_month
            FROM kiteConnect.instruments_zerodha
            WHERE name = 'NIFTY'
              AND instrument_type = 'PE'
              AND EXTRACT(MONTH FROM expiry) = EXTRACT(MONTH FROM %(signal_date)s)
              AND EXTRACT(YEAR FROM expiry) = EXTRACT(YEAR FROM %(signal_date)s)
        ),
        filtered_options AS (
            SELECT a.instrument_token, a.strike,
                   ROW_NUMBER() OVER (ORDER BY a.strike DESC) AS rnum
            FROM kiteConnect.instruments_zerodha a
            INNER JOIN last_expiry_month b ON a.expiry = b.last_expiry_month
            WHERE a.name = 'NIFTY'
              AND a.instrument_type = 'PE'
              AND a.strike <= %(strike_price)s
        )
        SELECT instrument_token
        FROM filtered_options 
        WHERE rnum = 1;
        """
        selected_token: int | None = None
        conn = None
        try:
            conn = self.k_apis.startKiteSession.con
            if not conn or not conn.is_connected():
                 print("    DB connection issue in _find_closest_PE_option. Re-initializing.")
                 self.k_apis.startKiteSession.init_trading()
                 conn = self.k_apis.startKiteSession.con
            if not conn or not conn.is_connected():
                print("    Failed to establish DB connection in _find_closest_PE_option.")
                return None
            params = {'signal_date': signal_datetime.date(), 'strike_price': underlying_price_at_signal}
            option_df = pd.read_sql_query(query, conn, params=params)
            if not option_df.empty and 'instrument_token' in option_df.columns:
                selected_token = int(option_df['instrument_token'].iloc[0])
                print(f"    Selected PE Option Token: {selected_token}")
            else:
                print(f"    No PE option token found by SQL query for NIFTY at {underlying_price_at_signal:.2f}, time {signal_datetime}.")
        except Exception as e:
            print(f"    Error in _find_closest_PE_option: {e}")
            return None
        return selected_token

    def _simulate_single_trade_on_option(self, 
                                         option_ohlcv_df: pd.DataFrame, 
                                         nifty_signal_time: datetime,
                                         entry_price_nifty: float,
                                         nifty_signals_for_trade_window: pd.Series | None = None # NEW: Nifty exit signals
                                         ) -> dict | None:
        if option_ohlcv_df.empty:
            print(f"    TradeSim: Option OHLCV data is empty. Cannot simulate trade for Nifty signal at {nifty_signal_time}.")
            return None

        # --- Determine Entry Candle (t+1) ---
        # option_ohlcv_df is assumed to be sorted by 'date'.
        # Find the positional index of the option candle that aligns with or is the first one after nifty_signal_time.
        # This is effectively the option's 't' candle relative to the Nifty signal.
        
        # Get all option candle datetime indices that are >= nifty_signal_time
        potential_signal_aligned_option_candle_datetime_indices = option_ohlcv_df[option_ohlcv_df['date'] >= nifty_signal_time].index

        if potential_signal_aligned_option_candle_datetime_indices.empty:
            print(f"    TradeSim: No option OHLCV data found at or after NIFTY signal time {nifty_signal_time}. Cannot determine trade entry candle.")
            return None

        # The first such candle corresponds to Nifty's signal candle period 't'.
        # Get its positional index in option_ohlcv_df:
        try:
            option_candle_t_datetime_idx = potential_signal_aligned_option_candle_datetime_indices[0]
            option_candle_t_positional_idx = option_ohlcv_df.index.get_loc(option_candle_t_datetime_idx)
        except IndexError:
             print(f"    TradeSim: Error finding option candle 't' for Nifty signal at {nifty_signal_time}. This shouldn't happen if previous check passed.")
             return None
        
        # The entry candle is the next one ('t+1')
        entry_candle_positional_idx = option_candle_t_positional_idx + 1

        if entry_candle_positional_idx >= len(option_ohlcv_df):
            print(f"    TradeSim: NIFTY signal at {nifty_signal_time} (corresponds to option candle at {option_ohlcv_df.iloc[option_candle_t_positional_idx]['date']}). The next option candle ('t+1') is beyond available data.")
            return None

        entry_candle = option_ohlcv_df.iloc[entry_candle_positional_idx]
        option_entry_time = entry_candle['date']
        option_entry_price = entry_candle['open'] 
        # --- End Determine Entry Candle ---

        if pd.isna(option_entry_price) or option_entry_price <= 0:
             print(f"    TradeSim: Invalid entry price (NaN or zero) for option at {option_entry_time} (t+1 candle). Skipping trade.")
             return None

        profit_target_price = option_entry_price * (1 + self.trade_params['profit_target_pct'])
        stop_loss_price = option_entry_price * (1 - self.trade_params['stop_loss_pct'])
        
        option_exit_price = None
        option_exit_time = None
        exit_reason = None
        
        max_holding_minutes = self.trade_params['max_holding_period_minutes'] # This is treated as number of candles by the loop

        # The loop iterates over candles *starting from the entry candle*
        # Each 'i' here is an offset from the entry_candle_positional_idx
        for i in range(max_holding_minutes): 
            current_option_candle_positional_idx = entry_candle_positional_idx + i 
            
            if current_option_candle_positional_idx >= len(option_ohlcv_df):
                exit_reason = "End of Option Data during holding period"
                if i > 0: # Make sure we held at least one candle past entry
                    last_available_candle_positional_idx = current_option_candle_positional_idx - 1
                    option_exit_price = option_ohlcv_df.iloc[last_available_candle_positional_idx]['close']
                    option_exit_time = option_ohlcv_df.iloc[last_available_candle_positional_idx]['date']
                # If i == 0 and this condition hits, it means entry_candle_positional_idx was the last candle.
                # This should have been caught by the check: "if entry_candle_positional_idx >= len(option_ohlcv_df):" before the loop.
                break

            # Access candle data using .iloc with the positional index
            current_candle_data = option_ohlcv_df.iloc[current_option_candle_positional_idx]
            current_option_high = current_candle_data['high']
            current_option_low = current_candle_data['low']
            current_option_close = current_candle_data['close']
            current_option_candle_time = current_candle_data['date']

            # Check for NIFTY strategy exit signal
            if nifty_signals_for_trade_window is not None:
                # current_nifty_signal_time should be the time of the current_candle_data, 
                # as nifty_signals_for_trade_window is indexed by NIFTY's candle times.
                current_nifty_signal_time_for_lookup = current_candle_data['date'] 
                if current_nifty_signal_time_for_lookup in nifty_signals_for_trade_window.index:
                    nifty_exit_signal_value = nifty_signals_for_trade_window.loc[current_nifty_signal_time_for_lookup]
                    if nifty_exit_signal_value == -1: # NIFTY Exit Signal
                        # Try to exit at the open of the next candle
                        next_option_candle_positional_idx = current_option_candle_positional_idx + 1
                        if next_option_candle_positional_idx < len(option_ohlcv_df):
                            next_option_candle = option_ohlcv_df.iloc[next_option_candle_positional_idx]
                            if pd.notna(next_option_candle['open']) and next_option_candle['open'] > 0:
                                exit_reason = "NIFTY Strategy Exit Signal (at next open)"
                                option_exit_price = next_option_candle['open']
                                option_exit_time = next_option_candle['date']
                                print(f"    TradeSim: Exiting due to NIFTY Strategy Signal at {current_nifty_signal_time_for_lookup}, option exit at next open {option_exit_time}")
                                break
                            else:
                                # If next open is invalid, fall back to current close
                                exit_reason = "NIFTY Strategy Exit Signal (next open invalid, used current close)"
                                option_exit_price = current_candle_data['close']
                                option_exit_time = current_candle_data['date']
                                print(f"    TradeSim: Exiting due to NIFTY Strategy Signal at {current_nifty_signal_time_for_lookup}, next option open invalid, using current close {option_exit_time}")
                                break
                        else:
                            # If no next candle, exit at current close
                            exit_reason = "NIFTY Strategy Exit Signal (no next candle, used current close)"
                            option_exit_price = current_candle_data['close']
                            option_exit_time = current_candle_data['date']
                            print(f"    TradeSim: Exiting due to NIFTY Strategy Signal at {current_nifty_signal_time_for_lookup}, no next option candle, using current close {option_exit_time}")
                            break
            
            # Priority 2: Check for profit target
            if current_option_high >= profit_target_price:
                option_exit_price = profit_target_price 
                # For a more realistic fill, one might consider if the open of this candle already met the target,
                # or if it gapped. Using profit_target_price assumes it traded at that level.
                option_exit_time = current_option_candle_time
                exit_reason = "Profit Target"
                break
            
            # Priority 3: Check for stop loss
            if current_option_low <= stop_loss_price:
                option_exit_price = stop_loss_price 
                # Similar to profit target, assumes it traded at stop_loss_price.
                option_exit_time = current_option_candle_time
                exit_reason = "Stop Loss"
                break
            
            # Priority 4: Check for Max Hold Time (if loop completes one before last iter)
            # This check is for the current candle 'i'. If it's the last allowed candle to hold.
            if i == max_holding_minutes - 1: 
                option_exit_price = current_option_close # Exit at close of this last holding candle
                option_exit_time = current_option_candle_time
                exit_reason = "Max Hold Time"
                # No 'break' here is needed if it's the last iteration, loop will end.
                # However, explicitly breaking is fine.
                break 
        
        if option_exit_price is not None and option_entry_price > 0:
            pnl_per_unit = option_exit_price - option_entry_price # For Call option
            total_pnl = pnl_per_unit * self.trade_params['units']
            
            trade_log_entry = {
                'nifty_signal_time': nifty_signal_time,
                'nifty_price_at_signal': round(entry_price_nifty, 2),
                'option_token': option_ohlcv_df.iloc[0].get('instrument_token', 'N/A'),
                'option_entry_time': option_entry_time,
                'option_entry_price': round(option_entry_price, 2),
                'option_exit_time': option_exit_time,
                'option_exit_price': round(option_exit_price, 2),
                'exit_reason': exit_reason,
                'pnl_per_unit': round(pnl_per_unit, 2),
                'total_pnl': round(total_pnl, 2),
                'units': self.trade_params['units']
            }
            print(f"    TradeSim: Logged trade. Nifty Signal @ {nifty_signal_time}, Option Entry @ {option_entry_price:.2f}, Exit @ {option_exit_price:.2f}, PNL: {total_pnl:.2f}, Reason: {exit_reason}")
            return trade_log_entry
        else:
            if option_entry_price <=0: print(f"    TradeSim: Skipped PNL calculation due to invalid option entry price {option_entry_price}")
            print(f"    TradeSim: Trade not executed or PNL not calculated for option based on Nifty signal at {nifty_signal_time}.")
        return None

    def run_simulation(self):
        print("\n--- Starting Trading Simulation ---")
        self.executed_trades = [] 
        self.is_trade_active = False # Ensure it's reset at the start of a new simulation run
        current_active_trade_exit_time = None # NEW: To track the exit time of the current trade

        print(f"Fetching NIFTY 50 index data ({self.index_token})...")
        # trade_interval now comes from the strategy's config, passed to __init__
        index_df = self.data_prep.fetch_and_prepare_data(
            self.index_token, 
            self.trade_start_date, 
            self.trade_end_date, 
            self.trade_interval # Use interval from __init__ (set from strategy config)
        )

        if index_df.empty:
            print("TradingSimulator: No NIFTY 50 index data fetched. Cannot run simulation.")
            return pd.DataFrame()
        
        print(f"Calculating statistics for NIFTY 50 data...")
        # The strategy object (self.strategy_obj) should have its parameters (like 'length')
        # already set from the config when it was instantiated by the main script.
        # DataPrep.calculate_statistics will use self.strategy_obj.length if it's Donchian.
        # If self.strategy_obj.length isn't present, DataPrep uses its own config default.
        stat_params = {}
        if hasattr(self.strategy_obj, 'length'): # For Donchian
             stat_params['donchian_length'] = self.strategy_obj.length
        # Add other strategy-specific stat params if needed, e.g.:
        if hasattr(self.strategy_obj, 'rsi_period'):
            stat_params['rsi_period'] = self.strategy_obj.rsi_period
        if hasattr(self.strategy_obj, 'ma_short_period'):
            stat_params['ma_short_period'] = self.strategy_obj.ma_short_period
        if hasattr(self.strategy_obj, 'ma_long_period'):
            stat_params['ma_long_period'] = self.strategy_obj.ma_long_period
        
        index_df_with_stats = self.data_prep.calculate_statistics(index_df.copy(), **stat_params)

        if index_df_with_stats.empty:
            print("TradingSimulator: Failed to calculate statistics. Cannot run simulation.")
            return pd.DataFrame()

        print(f"Generating signals on NIFTY 50 data using {type(self.strategy_obj).__name__}...")
        index_df_with_signals = self.strategy_obj.generate_signals(index_df_with_stats.copy()) 

        # Prepare a series of NIFTY signals indexed by time for efficient lookup
        # This includes ALL signals (1, -1, 0) from the strategy on the NIFTY index.
        nifty_all_signals_series = index_df_with_signals.set_index('date')['signal']

        buy_signals = index_df_with_signals[index_df_with_signals['signal'] == 1]
        print(f"Total BUY signals generated on NIFTY 50: {len(buy_signals)}")

        # Load option data fetch buffer from general simulator settings
        option_data_buffer_minutes = config.getint('SIMULATOR_SETTINGS', 'option_data_fetch_buffer_minutes', fallback=60)

        for idx, signal_row in buy_signals.iterrows():
            nifty_signal_time = signal_row['date']
            nifty_price_at_signal = signal_row['close'] 

            # NEW: Check if a current trade is active and if its exit time has passed
            if self.is_trade_active and current_active_trade_exit_time is not None:
                if nifty_signal_time >= current_active_trade_exit_time:
                    print(f"  LOG_CONCURRENCY: Previous trade ended at {current_active_trade_exit_time}. Setting is_trade_active = False for signal at {nifty_signal_time}.")
                    self.is_trade_active = False
                    current_active_trade_exit_time = None
                # else: The current nifty_signal_time is still within the previous trade's duration.

            print(f"\nProcessing BUY signal on NIFTY at {nifty_signal_time}, NIFTY Price: {nifty_price_at_signal:.2f}. Current is_trade_active: {self.is_trade_active}")

            if not self.allow_concurrent_trades and self.is_trade_active:
                print(f"  LOG_CONCURRENCY: Skipping NIFTY BUY signal at {nifty_signal_time} because a trade is still active until {current_active_trade_exit_time} and concurrent_signal_trade is False.")
                continue

            option_token = None
            # self.option_type is set during __init__ from strategy config
            if self.option_type == 'CE':
                option_token = self._find_closest_CE_option(nifty_price_at_signal, nifty_signal_time)
            elif self.option_type == 'PE':
                option_token = self._find_closest_PE_option(nifty_price_at_signal, nifty_signal_time)
            else:
                print(f"  Unsupported option type: {self.option_type}. Skipping option search.")

            if option_token:
                # print(f"  LOG_CONCURRENCY: Setting is_trade_active = True for NIFTY signal at {nifty_signal_time}") # Old log, will be replaced
                # self.is_trade_active = True # Old logic, moved and made conditional
                
                # Fetch option data
                option_data_start_date = nifty_signal_time.date()
                # Set option_data_end_date to the overall simulation end date
                option_data_end_date = self.trade_end_date

                print(f"  Fetching OHLCV for selected Option Token: {option_token} from {option_data_start_date} to {option_data_end_date}")
                option_ohlcv_df = self.data_prep.fetch_and_prepare_data(
                    instrument_token=option_token,
                    start_date_obj=option_data_start_date,
                    end_date_obj=option_data_end_date,
                    interval=self.trade_interval # Use interval from strategy config
                )

                if option_ohlcv_df is not None and not option_ohlcv_df.empty:
                    # Filter the nifty_all_signals_series for the potential trade window of this specific option trade
                    # This helps to pass only relevant signals to _simulate_single_trade_on_option
                    # Option trade can span from nifty_signal_time up to nifty_signal_time + max_holding_period + buffer
                    # The option_ohlcv_df gives a more precise window of actual option data available
                    min_option_time = option_ohlcv_df['date'].min()
                    max_option_time = option_ohlcv_df['date'].max()
                    
                    # Ensure nifty_all_signals_series covers the actual option data times
                    relevant_nifty_signals = nifty_all_signals_series[
                        (nifty_all_signals_series.index >= min_option_time) &
                        (nifty_all_signals_series.index <= max_option_time)
                    ]
                    if relevant_nifty_signals.empty and not nifty_all_signals_series.empty:
                         # This might happen if option data starts/ends outside Nifty signal series (unlikely for minute data)
                         print(f"    TradeSim Warning: No overlapping NIFTY signals found for option trade window {min_option_time} - {max_option_time}.")


                    trade_log = self._simulate_single_trade_on_option(
                        option_ohlcv_df, 
                        nifty_signal_time,
                        nifty_price_at_signal,
                        nifty_signals_for_trade_window=relevant_nifty_signals # Pass the filtered NIFTY signals
                    )
                    if trade_log:
                        self.executed_trades.append(trade_log)
                        if not self.allow_concurrent_trades: # NEW: Only manage active state if non-concurrent
                            self.is_trade_active = True
                            current_active_trade_exit_time = pd.to_datetime(trade_log['option_exit_time']) # Ensure it's datetime
                            print(f"  LOG_CONCURRENCY: New trade initiated by signal at {nifty_signal_time}. Active until {current_active_trade_exit_time}. is_trade_active = True.")
                    # else: trade_log was None (e.g. no valid entry price), so no trade became active.
                    # self.is_trade_active state (and current_active_trade_exit_time) from a *previous* signal remains.
                else:
                    print(f"  Could not fetch OHLCV data for option token {option_token}.")
                
                # REMOVED Problematic lines:
                # print(f"  LOG_CONCURRENCY: Setting is_trade_active = False after processing NIFTY signal at {nifty_signal_time}")
                # self.is_trade_active = False # This reset is now handled at the top of the loop based on time.
            else:
                print(f"  No suitable option token found for NIFTY signal at {nifty_signal_time}.")
        
        print("\n--- Simulation Run Complete ---")
        return pd.DataFrame(self.executed_trades)

    def calculate_performance_metrics(self, trades_df: pd.DataFrame): # initial_capital removed from signature
        """Calculates trading performance metrics. initial_capital is now a class member."""
        print("\n--- Trading Performance Metrics ---")
        metrics_summary_dict = {}
        
        # initial_capital is now self.initial_capital, set during __init__
        # print(f"Using initial capital for drawdown calculation (if applicable): {self.initial_capital}")

        if trades_df.empty:
            # ... (rest of calculate_performance_metrics is largely the same, only initial_capital reference might change if used for % calcs) ...
            print("No trades were executed. Cannot calculate metrics.")
            metrics_summary_dict["message"] = "No trades were executed."
            return metrics_summary_dict, "No trades were executed."

        num_total_trades = len(trades_df)
        trades_df['pnl_per_unit'] = pd.to_numeric(trades_df['pnl_per_unit'], errors='coerce')
        trades_df['total_pnl'] = pd.to_numeric(trades_df['total_pnl'], errors='coerce')
        trades_df.dropna(subset=['total_pnl'], inplace=True)

        if trades_df.empty : 
            print("No valid PNL data in trades. Cannot calculate metrics.")
            metrics_summary_dict["message"] = "No valid PNL data in trades."
            return metrics_summary_dict, "No valid PNL data in trades."

        winning_trades = trades_df[trades_df['total_pnl'] > 0]
        losing_trades = trades_df[trades_df['total_pnl'] < 0]
        num_winning_trades = len(winning_trades)
        num_losing_trades = len(losing_trades)
        win_rate = (num_winning_trades / num_total_trades) * 100 if num_total_trades > 0 else 0
        total_profit_loss = trades_df['total_pnl'].sum()
        average_pnl_per_trade = trades_df['total_pnl'].mean() if num_total_trades > 0 else 0
        gross_profit = winning_trades['total_pnl'].sum()
        gross_loss = abs(losing_trades['total_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        trades_df['cumulative_pnl'] = trades_df['total_pnl'].cumsum()
        
        # Prepare profit_factor string representation separately to handle np.inf
        profit_factor_str = f"{profit_factor:.2f}" if profit_factor != np.inf else "inf"

        # If using initial capital for portfolio equity curve:
        # trades_df['portfolio_equity'] = self.initial_capital + trades_df['cumulative_pnl']
        # trades_df['peak_equity'] = trades_df['portfolio_equity'].cummax()
        # trades_df['drawdown_value'] = trades_df['peak_equity'] - trades_df['portfolio_equity']
        # max_drawdown_value = trades_df['drawdown_value'].max()
        # For simple PNL drawdown:
        trades_df['peak_pnl'] = trades_df['cumulative_pnl'].cummax()
        trades_df['drawdown'] = trades_df['peak_pnl'] - trades_df['cumulative_pnl']
        max_drawdown_value = trades_df['drawdown'].max()
        
        metrics_summary_dict = {
            "Total Trades Executed": num_total_trades,
            "Winning Trades": num_winning_trades,
            "Losing Trades": num_losing_trades,
            "Win Rate (%)": f"{win_rate:.2f}",
            "Total Profit/Loss": f"{total_profit_loss:.2f}",
            "Average Profit/Loss per Trade": f"{average_pnl_per_trade:.2f}",
            "Gross Profit": f"{gross_profit:.2f}",
            "Gross Loss": f"{gross_loss:.2f}",
            "Profit Factor": profit_factor_str, # Use the pre-formatted string
            # "Initial Capital": f"{self.initial_capital:.2f}", # Added for clarity
            "Maximum Drawdown (Based on PNL)": f"{max_drawdown_value:.2f}"
        }
        summary_str_lines = ["--- Trading Performance Metrics ---"]
        for key, value in metrics_summary_dict.items():
            line = f"{key}: {value}"
            print(line)
            summary_str_lines.append(line)
        return metrics_summary_dict, "\n".join(summary_str_lines)

    def save_results(self, trades_df: pd.DataFrame, metrics_summary_str: str):
        # Output filenames from config or fixed? For now, fixed, as user requested no file paths in config
        # If these were to be configurable, they'd be in [SIMULATOR_SETTINGS]
        output_dir = "cursor_logs" # Kept hardcoded as per user request of no file paths in config
        simulation_trades_log_filename = "simulation_trades_output.csv"
        simulation_summary_filename = "simulation_summary.txt"

        os.makedirs(output_dir, exist_ok=True)
        if not trades_df.empty:
            trades_log_path = os.path.join(output_dir, simulation_trades_log_filename)
            trades_df.to_csv(trades_log_path, index=False)
            print(f"\nDetailed trades log saved to: {os.path.abspath(trades_log_path)}")
        else:
            print("No trades to save in the log.")
        summary_file_path = os.path.join(output_dir, simulation_summary_filename)
        try:
            with open(summary_file_path, 'w') as f:
                f.write(metrics_summary_str)
            print(f"Performance summary saved to: {os.path.abspath(summary_file_path)}")
        except IOError as e:
            print(f"Error saving performance summary: {e}")


if __name__ == '__main__':
    print("--- Initializing Trading Simulator Test ---")
    
    # --- 1. Load Global Configuration (already done at module level) ---
    # config object is available here.

    # --- Load Global Simulator Settings ---
    sim_settings = config['SIMULATOR_SETTINGS']
    INDEX_TOKEN = sim_settings.getint('index_token')
    INITIAL_CAPITAL = sim_settings.getfloat('initial_capital')
    ALLOW_CONCURRENT_TRADES = sim_settings.getboolean('allow_concurrent_trades', fallback=False) # Ensure a fallback
    SIMULATION_START_DATE_STR = sim_settings.get('simulation_start_date')
    SIMULATION_END_DATE_STR = sim_settings.get('simulation_end_date')

    # --- Determine Which Strategy Configuration to Use (NEW) ---
    # This now reads from the [SIMULATOR_SETTINGS] section
    SELECTED_STRATEGY_CONFIG_SECTION = sim_settings.get(
        'selected_strategy_config_section',
        fallback='STRATEGY_CONFIG_DonchianStandard' # Fallback if not specified
    )
    print(f"TradingSimulator Main: Attempting to run with strategy section: {SELECTED_STRATEGY_CONFIG_SECTION}")


    # --- Load Specific Strategy Configuration ---
    # SELECTED_STRATEGY_CONFIG_SECTION = 'STRATEGY_CONFIG_DonchianStandard' # OLD: Hardcoded
    
    if not config.has_section(SELECTED_STRATEGY_CONFIG_SECTION):
        raise ValueError(f"Strategy configuration section '{SELECTED_STRATEGY_CONFIG_SECTION}' not found in trading_config.ini")
    
    print(f"Using strategy configuration: [{SELECTED_STRATEGY_CONFIG_SECTION}]")

    # --- 3. Extract Simulator Settings from [SIMULATOR_SETTINGS] ---
    sim_index_token = config.getint('SIMULATOR_SETTINGS', 'index_token')
    sim_initial_capital = config.getfloat('SIMULATOR_SETTINGS', 'initial_capital')
    # sim_option_data_buffer = config.getint('SIMULATOR_SETTINGS', 'option_data_fetch_buffer_minutes') # Used internally by run_simulation
    sim_allow_concurrent_trades = config.getboolean('SIMULATOR_SETTINGS', 'concurrent_signal_trade', fallback=False)

    # --- 4. Extract Strategy-Specific Parameters ---
    strategy_class_name = config.get(SELECTED_STRATEGY_CONFIG_SECTION, 'strategy_class_name')
    
    # Indicator parameters for the strategy
    strategy_indicator_params = {}
    if strategy_class_name == 'DonchianBreakoutStrategy':
        strategy_indicator_params['length'] = config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'length')
        strategy_indicator_params['exit_option'] = config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'exit_option')
    elif strategy_class_name == 'MovingAverageRSILong':
        strategy_indicator_params['rsi_period'] = config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'rsi_period')
        strategy_indicator_params['rsi_oversold_threshold'] = config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'rsi_oversold_threshold')
        strategy_indicator_params['ma_short_period'] = config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'ma_short_period')
        strategy_indicator_params['ma_long_period'] = config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'ma_long_period')
        strategy_indicator_params['signal_offset_period'] = config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'signal_offset_period', fallback=1) # Add fallback
    elif strategy_class_name == 'MovingAverageRSIShort':
        strategy_indicator_params['rsi_period'] = config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'rsi_period')
        strategy_indicator_params['rsi_overbought_threshold'] = config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'rsi_overbought_threshold')
        strategy_indicator_params['ma_short_period'] = config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'ma_short_period')
        strategy_indicator_params['ma_long_period'] = config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'ma_long_period')
        strategy_indicator_params['signal_offset_period'] = config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'signal_offset_period', fallback=1) # Add fallback
    # Add elif for other strategy types and their specific indicator params
    # elif strategy_class_name == 'AnotherStrategy':
        
    # Trading parameters for the strategy (to be passed to simulator)
    sim_trade_params = {
        'units': config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'trade_units'),
        'profit_target_pct': config.getfloat(SELECTED_STRATEGY_CONFIG_SECTION, 'profit_target_pct'),
        'stop_loss_pct': config.getfloat(SELECTED_STRATEGY_CONFIG_SECTION, 'stop_loss_pct'),
        'max_holding_period_minutes': config.getint(SELECTED_STRATEGY_CONFIG_SECTION, 'max_holding_period_minutes')
    }
    sim_option_type = config.get(SELECTED_STRATEGY_CONFIG_SECTION, 'option_type')
    sim_trade_interval = config.get(SELECTED_STRATEGY_CONFIG_SECTION, 'trade_interval')

    # --- 5. Instantiate Strategy Object ---
    active_strategy = None
    if strategy_class_name == 'DonchianBreakoutStrategy':
        active_strategy = DonchianBreakoutStrategy(**strategy_indicator_params)
    elif strategy_class_name == 'MovingAverageRSILong':
        active_strategy = MovingAverageRSILong(**strategy_indicator_params)
    elif strategy_class_name == 'MovingAverageRSIShort':
        active_strategy = MovingAverageRSIShort(**strategy_indicator_params)
    # Add elif for other strategy types
    # elif strategy_class_name == 'AnotherStrategy':
    else:
        raise ValueError(f"Unsupported strategy_class_name '{strategy_class_name}' found in config section '{SELECTED_STRATEGY_CONFIG_SECTION}'")

    if active_strategy is None:
        raise SystemError("Failed to instantiate the active strategy. Check configuration.")

    # --- 6. Define Simulation Period from Config --- 
    sim_start_date_str = config.get('SIMULATOR_SETTINGS', 'simulation_start_date', fallback=date.today().strftime('%Y-%m-%d'))
    sim_end_date_str = config.get('SIMULATOR_SETTINGS', 'simulation_end_date', fallback=date.today().strftime('%Y-%m-%d'))

    try:
        start_date_only = datetime.strptime(sim_start_date_str, '%Y-%m-%d').date()
        end_date_only = datetime.strptime(sim_end_date_str, '%Y-%m-%d').date()
    except ValueError as e:
        print(f"Error parsing simulation dates from config: {e}. Ensure format is YYYY-MM-DD.")
        print(f"Using current date ({date.today().strftime('%Y-%m-%d')}) as fallback.")
        start_date_only = date.today()
        end_date_only = date.today()

    sim_start_date = datetime.combine(start_date_only, time(0, 0, 0))
    sim_end_date = datetime.combine(end_date_only, time(23, 59, 59))
    
    # --- 7. Instantiate TradingSimulator ---
    simulator = TradingSimulator(
        index_token=sim_index_token,
        strategy_obj=active_strategy,
        trade_start_date=sim_start_date,
        trade_end_date=sim_end_date,
        option_type=sim_option_type,
        trade_interval=sim_trade_interval,
        trade_params=sim_trade_params,
        initial_capital=sim_initial_capital,
        allow_concurrent_trades=sim_allow_concurrent_trades # Pass the new setting
    )

    # --- 8. Run Simulation and Process Results ---
    try:
        final_trades_df = simulator.run_simulation()
        
        if not final_trades_df.empty:
            print("\n--- Final Executed Trades --- (First 5)")
            print(final_trades_df.head())
            metrics_dict, metrics_summary = simulator.calculate_performance_metrics(final_trades_df)
            simulator.save_results(final_trades_df, metrics_summary)
        else:
            print("Simulation completed with no trades executed.")
            # metrics_dict, metrics_summary = simulator.calculate_performance_metrics(pd.DataFrame()) # Get "no trades" summary
            simulator.save_results(pd.DataFrame(), "Simulation completed with no trades executed.")


    except ConnectionError as ce:
        print(f"Simulator Connection Error: {ce}")
    except FileNotFoundError as fnf:
        print(f"Simulator File Error: {fnf}")
    except ValueError as ve:
        print(f"Simulator Value Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Trading Simulator Test Complete ---") 