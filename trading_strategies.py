import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from myKiteLib import kiteAPIs # Assuming myKiteLib.py is in the same directory or accessible in PYTHONPATH
from datetime import date, datetime, timedelta # For type hinting and date conversions
import configparser # Added for reading config file
import os # Added for file path operations

class DataPrep:
    """
    Handles fetching and preparing historical market data for strategies.
    """
    def __init__(self):
        """
        Initializes the DataPrep class by creating an instance of kiteAPIs.
        """
        try:
            self.k_apis = kiteAPIs()
            print("DataPrep: kiteAPIs initialized successfully.")
        except ImportError:
            print("DataPrep Error: myKiteLib.py or kiteAPIs class not found. Ensure it's in the correct path.")
            self.k_apis = None
        except Exception as e:
            print(f"DataPrep Error: Could not initialize kiteAPIs: {e}")
            self.k_apis = None

    def _parse_interval_string(self, interval_str: str) -> int:
        """Converts interval string like 'minute', '5minute' to integer minutes."""
        if isinstance(interval_str, int):
            return interval_str # Already an int
        if not isinstance(interval_str, str):
            print(f"DataPrep Warning: Interval '{interval_str}' is not a string. Defaulting to 1 minute.")
            return 1
        
        interval_str_lower = interval_str.lower()
        if interval_str_lower == 'minute' or interval_str_lower == '1minute':
            return 1
        elif interval_str_lower == '60minute' or interval_str_lower == 'hour' or interval_str_lower == '1hour' or interval_str_lower == '60min':
            return 60
        elif 'minute' in interval_str_lower:
            try:
                return int(interval_str_lower.replace('minute', ''))
            except ValueError:
                print(f"DataPrep Warning: Could not parse numeric value from interval '{interval_str}'. Defaulting to 1 minute.")
                return 1
        else:
            try:
                # Attempt to parse as int if it's just a number string like "5"
                return int(interval_str_lower)
            except ValueError:
                print(f"DataPrep Warning: Unknown interval format '{interval_str}'. Defaulting to 1 minute.")
                return 1

    def fetch_and_prepare_data(self, instrument_token: int, start_date_obj: date, end_date_obj: date, interval: str = 'minute', warm_up_days: int = 0) -> pd.DataFrame:
        """
        Fetches 1-minute historical data from the database via myKiteLib,
        then resamples it to the specified 'interval' if needed, and prepares it.
        Includes an optional warm-up period by fetching data prior to start_date_obj.

        Args:
            instrument_token: The instrument token to fetch data for.
            start_date_obj: The intended start date for the simulation/analysis period.
            end_date_obj: The end date for data fetching (datetime.date object).
            interval: The target candle interval (e.g., 'minute', '5minute', '15minute').
            warm_up_days: Number of extra days of data to fetch before start_date_obj for warm-up.

        Returns:
            A pandas DataFrame with prepared OHLCV data at the target interval,
            sorted by date, with a 'date' column (datetime), and numeric OHLCV columns.
            Returns an empty DataFrame if fetching or preparation fails.
        """
        if not self.k_apis:
            print("DataPrep Error: kiteAPIs not initialized. Cannot fetch data.")
            return pd.DataFrame()

        actual_fetch_start_date = start_date_obj
        if warm_up_days > 0:
            actual_fetch_start_date = start_date_obj - timedelta(days=warm_up_days)
            print(f"DataPrep: Original start date {start_date_obj}, fetching from {actual_fetch_start_date} to include {warm_up_days} warm-up days.")

        start_date_str = actual_fetch_start_date.strftime('%Y-%m-%d')
        end_date_str = end_date_obj.strftime('%Y-%m-%d')

        # User requested to fetch as much data as needed.
        # For ZigZag with 5 points (XABCD), a considerable history might be needed.
        # Let's fetch a longer period (e.g., 1 year) before start_date_obj for robust ZigZag calculation,
        # then slice it if necessary, though the strategy will operate on the full fetched data.
        # The simulator passes trade_start_date and trade_end_date. We should honor these for the *simulation window*.
        # The strategy *itself* needs historical data *prior* to trade_start_date to form initial patterns.
        # This implies fetch_and_prepare_data should fetch data from an earlier point.
        # For now, we fetch what's requested by the simulator. The strategy must cope or this needs redesign.
        
        print(f"DataPrep: Fetching 1-minute data for token {instrument_token} from {start_date_str} to {end_date_str} (to be resampled to {interval})...")
        
        historical_df_minute = self.k_apis.extract_data_from_db(
            from_date=start_date_str,
            to_date=end_date_str,
            interval='minute', 
            instrument_token=instrument_token
        )

        if historical_df_minute is None or historical_df_minute.empty:
            print(f"DataPrep: No 1-minute data fetched for token {instrument_token} from {start_date_str} to {end_date_str}.")
            return pd.DataFrame()

        print(f"DataPrep: Successfully fetched {len(historical_df_minute)} rows of 1-minute data. Preparing...")

        if 'timestamp' in historical_df_minute.columns:
            historical_df_minute.rename(columns={'timestamp': 'date'}, inplace=True)
        
        if 'date' not in historical_df_minute.columns:
            print("DataPrep Critical Error: 'date' column not found in 1-minute data after potential rename.")
            return pd.DataFrame()
            
        historical_df_minute['date'] = pd.to_datetime(historical_df_minute['date'])
        
        cols_to_numeric = ['open', 'high', 'low', 'close', 'volume']
        for col in cols_to_numeric:
            if col in historical_df_minute.columns:
                historical_df_minute[col] = pd.to_numeric(historical_df_minute[col], errors='coerce')
            else:
                print(f"DataPrep Warning: Expected column '{col}' not found in 1-minute data for numeric conversion.")
        
        historical_df_minute.sort_values(by=['date'], inplace=True)
        historical_df_minute.drop_duplicates(subset=['date'], keep='first', inplace=True)

        target_interval_minutes = self._parse_interval_string(interval)

        if target_interval_minutes > 1:
            print(f"DataPrep: Resampling 1-minute data to {target_interval_minutes}-minute interval ('{interval}')...")
            if not hasattr(self.k_apis, 'convert_minute_data_interval'):
                print("DataPrep Error: kiteAPIs object does not have method 'convert_minute_data_interval'. Cannot resample.")
                print("DataPrep Warning: Proceeding with 1-minute data as resampling function is missing.")
                final_df = historical_df_minute
            else:
                try:
                    # Pass the integer interval (target_interval_minutes) 
                    # if k_apis.convert_minute_data_interval expects an integer.
                    final_df = self.k_apis.convert_minute_data_interval(historical_df_minute.copy(), to_interval=target_interval_minutes) # Pass INTEGER interval
                    if final_df is None or final_df.empty:
                        print(f"DataPrep Error: Resampling to '{interval}' ({target_interval_minutes} mins) resulted in empty data. Check resampling logic.")
                        return pd.DataFrame()
                    print(f"DataPrep: Resampled to {len(final_df)} rows at '{interval}' ({target_interval_minutes} mins) interval.")
                except Exception as e:
                    print(f"DataPrep Error during resampling to '{interval}' ({target_interval_minutes} mins): {e}")
                    print("DataPrep Warning: Proceeding with 1-minute data due to resampling error.")
                    final_df = historical_df_minute
        else:
            final_df = historical_df_minute

        # After resampling, check if the datetime column is named 'timestamp'
        if 'timestamp' in final_df.columns and 'date' not in final_df.columns:
            print("DataPrep: Found 'timestamp' column after resampling, renaming to 'date'.")
            final_df.rename(columns={'timestamp': 'date'}, inplace=True)
        elif 'date' not in final_df.columns:
            # If neither 'date' nor 'timestamp' is present, then we have an issue.
            # This also covers the case where the index might be datetime but not named 'date'.
            # A more robust solution might be needed if dates are in an unnamed index or other column.
            if isinstance(final_df.index, pd.DatetimeIndex):
                print("DataPrep: 'date' column not found, but DatetimeIndex is present. Resetting index.")
                final_df.reset_index(inplace=True)
                # If the reset index resulted in a column named 'index' or the original index name, try to rename to 'date'
                if final_df.columns[0] != 'date' : # Assuming date becomes the first column
                     # Check if the first column (potentially from reset_index) should be 'date'
                     # This is a simple heuristic. If index was named, it might take that name.
                    if final_df.index.name and final_df.index.name != 'date' and final_df.index.name in final_df.columns:
                        final_df.rename(columns={final_df.index.name: 'date'}, inplace=True)
                    elif 'index' in final_df.columns and 'date' not in final_df.columns: # Common if index was unnamed
                        final_df.rename(columns={'index': 'date'}, inplace=True)
                    else: # Fallback if first col is not 'date' yet
                         print(f"DataPrep: Attempting to rename first column '{final_df.columns[0]}' to 'date' after index reset.")
                         final_df.rename(columns={final_df.columns[0]: 'date'}, inplace=True)
            else:
                print(f"DataPrep Critical Error: 'date' or 'timestamp' column missing after resampling, and index is not DatetimeIndex. Columns: {final_df.columns.tolist()}, Index type: {type(final_df.index)}")
                return pd.DataFrame()
        
        # Ensure 'date' column is of datetime type after all manipulations
        if 'date' in final_df.columns:
            try:
                final_df['date'] = pd.to_datetime(final_df['date'])
                print("DataPrep: Ensured 'date' column is pd.to_datetime.")
            except Exception as e:
                print(f"DataPrep Critical Error: Failed to convert 'date' column to datetime: {e}. Column content head: {final_df['date'].head()}")
                return pd.DataFrame()
        else:
            print("DataPrep Critical Error: 'date' column still missing before final checks.")
            return pd.DataFrame()

        final_df.reset_index(drop=True, inplace=True) 
        required_cols = ['date', 'open', 'high', 'low', 'close'] 
        # Before dropna, ensure all required_cols are actually present after index handling
        missing_for_dropna = [col for col in required_cols if col not in final_df.columns]
        if missing_for_dropna:
            print(f"DataPrep Critical Error: Columns {missing_for_dropna} are missing before dropna.")
            return pd.DataFrame()
            
        final_df.dropna(subset=required_cols, inplace=True) # Drop rows where essential OHLC data might be NaN after resampling

        if not all(col in final_df.columns for col in required_cols):
            print(f"DataPrep Error: Missing one or more required columns {required_cols} after final preparation.")
            return pd.DataFrame()

        print(f"DataPrep: Data preparation complete for interval '{interval}'. Final shape: {final_df.shape}")
        return final_df

    def calculate_statistics(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculates technical indicators based on kwargs.
        For ZigZag strategy, this might not be used for primary indicators,
        as the strategy calculates ZigZag and patterns itself.
        This can be used for any other generic indicators if needed by a future strategy.
        """
        df = data.copy()
        print(f"DataPrep: calculate_statistics called. Kwargs: {kwargs} (Mostly bypassed for ZigZag)")
        return df

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals based on the input data.

        Args:
            data: DataFrame with OHLCV data, a 'date' column, 
                  and potentially other indicators. 
                  Expected columns: 'date', 'open', 'high', 'low', 'close'.

        Returns:
            DataFrame with an added 'signal' column.
            Signal values: 1 for BUY, -1 for SELL/EXIT, 0 for HOLD/NO_SIGNAL.
            May also include intermediate indicator columns used by the strategy.
        """
        pass

class TradingStrategy(BaseStrategy):
    """
    This is the main trading strategy class.
    Implement your custom trading logic in this class.
    """
    def __init__(self, kite_apis_instance: kiteAPIs, simulation_actual_start_date: date = None, **kwargs):
        """
        Initializes the TradingStrategy.

        Args:
            kite_apis_instance: An instance of the kiteAPIs class from myKiteLib.py.
            simulation_actual_start_date: The actual date from which trading simulation should start.
                                          Data before this date might be used for indicator warm-up.
            **kwargs: Strategy-specific parameters. 
                      Example: instrument_token=256265
        """
        self.strategy_name = "ZigZagHarmonicStrategy"
        print(f"Initialized {self.strategy_name}")

        if not isinstance(kite_apis_instance, kiteAPIs):
            raise TypeError("kite_apis_instance must be an instance of the kiteAPIs class.")
        self.k_apis = kite_apis_instance
        self.simulation_actual_start_date = simulation_actual_start_date
        if self.simulation_actual_start_date:
            print(f"  {self.strategy_name} actual simulation start date: {self.simulation_actual_start_date}")
        
        # Strategy parameters with defaults, potentially overridden by kwargs from config
        self.target01_ew_rate = float(kwargs.get('target01_ew_rate', 0.236))
        self.target01_tp_rate = float(kwargs.get('target01_tp_rate', 0.618))
        self.target01_sl_rate = float(kwargs.get('target01_sl_rate', -0.236))
        
        self.target02_active = str(kwargs.get('target02_active', 'false')).lower() == 'true'
        
        self.target02_ew_rate = float(kwargs.get('target02_ew_rate', 0.236))
        self.target02_tp_rate = float(kwargs.get('target02_tp_rate', 1.618))
        self.target02_sl_rate = float(kwargs.get('target02_sl_rate', -0.236))

        self.useAltTF = str(kwargs.get('usealttf', 'true')).lower() == 'true' # Adjusted key to lowercase
        self.altTF_interval_minutes = int(kwargs.get('alttf_interval_minutes', 60)) # Adjusted key to lowercase

        # Allow other kwargs (like instrument_token) to be set as attributes
        # and remove strategy-specific ones already processed to avoid overwriting with string versions
        processed_keys = [
            'target01_ew_rate', 'target01_tp_rate', 'target01_sl_rate',
            'target02_active', 'target02_ew_rate', 'target02_tp_rate', 'target02_sl_rate',
            'usealttf', 'alttf_interval_minutes'
        ]
        for key, value in kwargs.items():
            if key.lower() not in processed_keys: # Check lowercase to catch variations from config
                setattr(self, key, value)
                print(f"  {self.strategy_name} param set via kwargs: {key} = {value}")
            elif key.lower() in processed_keys and not hasattr(self, key):
                # This case handles if the key was in processed_keys but somehow not set above
                # (e.g. if default logic changes) or if user wants original string value for some reason.
                # For now, the specific setters above take precedence for type conversion.
                pass # Already handled with type conversion

        # Print critical parameters after they are set
        print(f"  {self.strategy_name} FINAL param: target01_ew_rate = {self.target01_ew_rate}")
        print(f"  {self.strategy_name} FINAL param: target01_tp_rate = {self.target01_tp_rate}")
        print(f"  {self.strategy_name} FINAL param: target01_sl_rate = {self.target01_sl_rate}")
        print(f"  {self.strategy_name} FINAL param: target02_active = {self.target02_active}")
        print(f"  {self.strategy_name} FINAL param: target02_ew_rate = {self.target02_ew_rate}")
        print(f"  {self.strategy_name} FINAL param: target02_tp_rate = {self.target02_tp_rate}")
        print(f"  {self.strategy_name} FINAL param: target02_sl_rate = {self.target02_sl_rate}")
        print(f"  {self.strategy_name} FINAL param: useAltTF = {self.useAltTF}")
        print(f"  {self.strategy_name} FINAL param: altTF_interval_minutes = {self.altTF_interval_minutes}")

        # Helper list for pattern column names in DataFrame
        self.pattern_names_for_df_columns = [ 
            'Bat', 'AntiBat', 'AltBat', 'Butterfly', 'AntiButterfly', 'ABCD', 
            'Gartley', 'AntiGartley', 'Crab', 'AntiCrab', 'Shark', 'AntiShark', 
            '5o', 'Wolf', 'HnS', 'ConTria', 'ExpTria'
        ]

        # --- Strategy State Variables ---
        self.active_long_trade = None
        self.active_short_trade = None
        self.last_pattern_info = None
        # Example for active_long_trade/active_short_trade:
        # {
        #    'type': 'long' / 'short',
        #    'pattern_name': 'Bull Bat',
        #    'entry_price': 100.50, # Actual entry price
        #    'initial_tp_price': 102.00, # TP at time of entry
        #    'initial_sl_price': 99.00,  # SL at time of entry
        #    'current_tp_price': 102.50, # Dynamically adjusted TP
        #    'current_sl_price': 98.50,  # Dynamically adjusted SL
        #    'target_level': 'T1' / 'T2',
        #    'c_price_entry': 101.0, # C price at time of entry pattern
        #    'd_price_entry': 99.5,  # D price at time of entry pattern
        #    'd_timestamp_entry': ..., # D timestamp of entry pattern
        #    'last_pivot_c_price_for_calc': 101.2, # C price used for latest TP/SL calc
        #    'last_pivot_d_price_for_calc': 99.8,  # D price used for latest TP/SL calc
        #    'last_pivot_d_timestamp_for_calc': ..., # D timestamp used for latest TP/SL calc
        #    'pattern_mode': 1, # 1 for bull, -1 for bear (of the entry pattern)
        #    'fib_tp_rate': 0.618, # Fib rate for TP for this trade
        #    'fib_sl_rate': -0.236  # Fib rate for SL for this trade
        # }

    def _calculate_zigzag_pivots(self, data: pd.DataFrame) -> list:
        """
        Calculates ZigZag pivot points (price and timestamp).
        Attempts to replicate the PineScript logic for `sz`.
        Returns a list of tuples: [(timestamp, price), ...]
        """
        n = len(data)
        if n < 2:
            return []

        sz_points_values = pd.Series([np.nan] * n, index=data.index)
        # Pine: _direction = _isUp[1] and _isDown ? -1 : _isDown[1] and _isUp ? 1 : nz(_direction[1])
        # nz(_direction[1]) means if _direction[1] is na, use 0 or last known non-na value.
        # For simplicity, let's track direction explicitly.
        
        pine_direction = 0.0 # 0: undetermined, 1: up, -1: down

        # Loop starts from index 1 as it uses previous bar data (i-1)
        for i in range(1, n):
            # Current candle properties
            open_curr = data['open'].iloc[i]
            close_curr = data['close'].iloc[i]
            high_curr = data['high'].iloc[i]
            low_curr = data['low'].iloc[i]
            
            # Previous candle properties
            open_prev = data['open'].iloc[i-1]
            close_prev = data['close'].iloc[i-1]
            high_prev = data['high'].iloc[i-1]
            low_prev = data['low'].iloc[i-1]

            isUp_prev = close_prev >= open_prev
            isDown_prev = close_prev <= open_prev
            
            isUp_curr = close_curr >= open_curr
            isDown_curr = close_curr <= open_curr

            # Update Pine direction state
            # Original: _direction = _isUp[1] and _isDown ? -1 : _isDown[1] and _isUp ? 1 : nz(_direction[1])
            # nz(_direction[1]) means use previous direction if no change pattern.
            prev_pine_direction_for_calc = pine_direction # This is nz(direction[1]) equivalent
            
            if isUp_prev and isDown_curr:
                pine_direction = -1
            elif isDown_prev and isUp_curr:
                pine_direction = 1
            # else: pine_direction remains its previous value (effect of nz)

            # Zigzag point logic from Pine:
            # _zigzag = _isUp[1] and _isDown and _direction[1] != -1 ? highest(2) : 
            #           _isDown[1] and _isUp and _direction[1] != 1 ? lowest(2) : na
            # _direction[1] refers to the direction *before* the current candle's influence changed it.
            # So, we use prev_pine_direction_for_calc.
            # highest(2) means max(high_curr, high_prev), lowest(2) means min(low_curr, low_prev)

            if isUp_prev and isDown_curr and prev_pine_direction_for_calc != -1:
                sz_points_values.iloc[i] = max(high_curr, high_prev)
            elif isDown_prev and isUp_curr and prev_pine_direction_for_calc != 1:
                sz_points_values.iloc[i] = min(low_curr, low_prev)
        
        # Extract non-NaN pivots
        actual_pivots = []
        for idx_val, price in sz_points_values.items(): # idx_val is the timestamp
            if not pd.isna(price):
                actual_pivots.append({'timestamp': idx_val, 'price': price})
        return actual_pivots

    # --- Pattern Recognition Helper Functions ---
    # Each takes X, A, B, C, D prices and mode (1 for bull, -1 for bear)
    def _check_direction(self, mode, c_price, d_price):
        if mode == 1: return d_price < c_price # Bullish: D is low, C is high
        if mode == -1: return d_price > c_price # Bearish: D is high, C is low
        return False

    def _isBat(self, xab, abc, bcd, xad, mode, c_price, d_price):
        _xab = xab >= 0.382 and xab <= 0.5
        _abc = abc >= 0.382 and abc <= 0.886
        _bcd = bcd >= 1.618 and bcd <= 2.618
        _xad = xad <= 0.618 # Matches Pine: xad <= 0.618 and xad <= 1.000 (effectively xad <= 0.618)
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isGartley(self, xab, abc, bcd, xad, mode, c_price, d_price):
        _xab = xab >= 0.5 and xab <= 0.618 # Pine uses 0.618, range can be 0.5-0.618
        _abc = abc >= 0.382 and abc <= 0.886
        _bcd = bcd >= 1.13 and bcd <= 2.618
        _xad = xad >= 0.75 and xad <= 0.875 # Pine is 0.786, range 0.75-0.875 covers it
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isABCD(self, xab, abc, bcd, xad, mode, c_price, d_price): # xab, xad not used
        _abc = abc >= 0.382 and abc <= 0.886
        _bcd = bcd >= 1.13 and bcd <= 2.618
        return _abc and _bcd and self._check_direction(mode, c_price, d_price)
    
    def _isAntiBat(self, xab, abc, bcd, xad, mode, c_price, d_price):
        _xab = xab >= 0.500 and xab <= 0.886    # Pine: 0.618
        _abc = abc >= 1.000 and abc <= 2.618    # Pine: 1.13 --> 2.618
        _bcd = bcd >= 1.618 and bcd <= 2.618    # Pine: 2.0  --> 2.618
        _xad = xad >= 0.886 and xad <= 1.000    # Pine: 1.13
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isAltBat(self, xab, abc, bcd, xad, mode, c_price, d_price):
        _xab = xab <= 0.382
        _abc = abc >= 0.382 and abc <= 0.886
        _bcd = bcd >= 2.0 and bcd <= 3.618
        _xad = xad <= 1.13
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isButterfly(self, xab, abc, bcd, xad, mode, c_price, d_price):
        _xab = xab <= 0.786
        _abc = abc >= 0.382 and abc <= 0.886
        _bcd = bcd >= 1.618 and bcd <= 2.618
        _xad = xad >= 1.27 and xad <= 1.618
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isAntiButterfly(self, xab, abc, bcd, xad, mode, c_price, d_price):
        _xab = xab >= 0.236 and xab <= 0.886    # Pine: 0.382 - 0.618
        _abc = abc >= 1.130 and abc <= 2.618    # Pine: 1.130 - 2.618
        _bcd = bcd >= 1.000 and bcd <= 1.382    # Pine: 1.27
        _xad = xad >= 0.500 and xad <= 0.886    # Pine: 0.618 - 0.786
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isAntiGartley(self, xab, abc, bcd, xad, mode, c_price, d_price):
        _xab = xab >= 0.500 and xab <= 0.886    # Pine: 0.618 -> 0.786
        _abc = abc >= 1.000 and abc <= 2.618    # Pine: 1.130 -> 2.618
        _bcd = bcd >= 1.500 and bcd <= 5.000    # Pine: 1.618
        _xad = xad >= 1.000 and xad <= 5.000    # Pine: 1.272
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isCrab(self, xab, abc, bcd, xad, mode, c_price, d_price):
        _xab = xab >= 0.500 and xab <= 0.875    # Pine: 0.886
        _abc = abc >= 0.382 and abc <= 0.886    
        _bcd = bcd >= 2.000 and bcd <= 5.000    # Pine: 3.618
        _xad = xad >= 1.382 and xad <= 5.000    # Pine: 1.618
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isAntiCrab(self, xab, abc, bcd, xad, mode, c_price, d_price):
        _xab = xab >= 0.250 and xab <= 0.500    # Pine: 0.276 -> 0.446
        _abc = abc >= 1.130 and abc <= 2.618    # Pine: 1.130 -> 2.618
        _bcd = bcd >= 1.618 and bcd <= 2.618    # Pine: 1.618 -> 2.618
        _xad = xad >= 0.500 and xad <= 0.750    # Pine: 0.618
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isShark(self, xab, abc, bcd, xad, mode, c_price, d_price):
        _xab = xab >= 0.500 and xab <= 0.875    # Pine: 0.5 --> 0.886
        _abc = abc >= 1.130 and abc <= 1.618    
        _bcd = bcd >= 1.270 and bcd <= 2.240    
        _xad = xad >= 0.886 and xad <= 1.130    # Pine: 0.886 --> 1.13
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isAntiShark(self, xab, abc, bcd, xad, mode, c_price, d_price):
        _xab = xab >= 0.382 and xab <= 0.875    # Pine: 0.446 --> 0.618
        _abc = abc >= 0.500 and abc <= 1.000    # Pine: 0.618 --> 0.886
        _bcd = bcd >= 1.250 and bcd <= 2.618    # Pine: 1.618 --> 2.618
        _xad = xad >= 0.500 and xad <= 1.250    # Pine: 1.130 --> 1.130
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _is5o(self, xab, abc, bcd, xad, mode, c_price, d_price): # Pine variable name is5o
        _xab = xab >= 1.13 and xab <= 1.618
        _abc = abc >= 1.618 and abc <= 2.24
        _bcd = bcd >= 0.5 and bcd <= 0.625 # Pine: 0.5
        _xad = xad >= 0.0 and xad <= 0.236 # Pine: negative? (Handled by check_direction and point ordering)
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isWolf(self, xab, abc, bcd, xad, mode, c_price, d_price): # Pine variable name isWolf
        _xab = xab >= 1.27 and xab <= 1.618
        _abc = abc >= 0 and abc <= 5 # Pine has very wide range
        _bcd = bcd >= 1.27 and bcd <= 1.618
        _xad = xad >= 0.0 and xad <= 5 # Pine has very wide range
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isHnS(self, xab, abc, bcd, xad, mode, c_price, d_price): # Pine variable name isHnS
        _xab = xab >= 2.0 and xab <= 10
        _abc = abc >= 0.90 and abc <= 1.1
        _bcd = bcd >= 0.236 and bcd <= 0.88
        _xad = xad >= 0.90 and xad <= 1.1
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isConTria(self, xab, abc, bcd, xad, mode, c_price, d_price): # Pine variable name isConTria
        _xab = xab >= 0.382 and xab <= 0.618
        _abc = abc >= 0.382 and abc <= 0.618
        _bcd = bcd >= 0.382 and bcd <= 0.618
        _xad = xad >= 0.236 and xad <= 0.764
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _isExpTria(self, xab, abc, bcd, xad, mode, c_price, d_price): # Pine variable name isExpTria
        _xab = xab >= 1.236 and xab <= 1.618
        _abc = abc >= 1.000 and abc <= 1.618
        _bcd = bcd >= 1.236 and bcd <= 2.000
        _xad = xad >= 2.000 and xad <= 2.236
        return _xab and _abc and _bcd and _xad and self._check_direction(mode, c_price, d_price)

    def _get_fib_level(self, c_price, d_price, rate, is_bullish_pattern):
        """Calculates Fibonacci level based on C, D points and rate."""
        fib_range = abs(d_price - c_price)
        if pd.isna(fib_range) or fib_range < 1e-6: # Avoid division by zero or tiny range
            return np.nan

        if is_bullish_pattern: # D is a low, C is a high (d_price < c_price usually for valid pattern)
            return d_price + (fib_range * rate)
        else: # Bearish pattern, D is a high, C is a low (d_price > c_price usually)
            return d_price - (fib_range * rate)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['signal'] = 0
        df['pattern_tag'] = ""
        df['zigzag_price'] = np.nan
        
        # Initialize new columns for detailed pattern analysis
        df['xab_ratio'] = np.nan
        df['abc_ratio'] = np.nan
        df['bcd_ratio'] = np.nan
        df['xad_ratio'] = np.nan
        
        # self.pattern_names_for_df_columns is initialized in __init__
        # but ensure it's available if __init__ was somehow skipped (e.g. direct test of generate_signals)
        if not hasattr(self, 'pattern_names_for_df_columns'):
             self.pattern_names_for_df_columns = [ 
                'Bat', 'AntiBat', 'AltBat', 'Butterfly', 'AntiButterfly', 'ABCD', 
                'Gartley', 'AntiGartley', 'Crab', 'AntiCrab', 'Shark', 'AntiShark', 
                '5o', 'Wolf', 'HnS', 'ConTria', 'ExpTria'
            ]

        for p_name in self.pattern_names_for_df_columns:
            df[f'is{p_name}_bull'] = pd.NA 
            df[f'is{p_name}_bear'] = pd.NA

        df['pattern_ew1_price'] = np.nan
        df['pattern_tp1_price'] = np.nan
        df['pattern_sl1_price'] = np.nan
        # If target02_active can be true, you might add pattern_ew2_price etc. here too

        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"{self.strategy_name}: DataFrame index is not DatetimeIndex. This is required.")
            # Attempt to set it if 'date' column exists, matching __main__ block
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    print(f"{self.strategy_name}: Successfully set 'date' column as DatetimeIndex.")
                except Exception as e:
                    print(f"{self.strategy_name}: Failed to set 'date' as DatetimeIndex: {e}. Returning empty signals.")
                    return df
            else:
                print(f"{self.strategy_name}: 'date' column not found to set as DatetimeIndex. Returning empty signals.")
                return df

        if len(df) < 10: # Increased minimum for 5-point patterns (XABCD requires at least 5 pivots)
            print(f"{self.strategy_name}: Data too short ({len(df)} rows). Needs more for ZigZag and pattern detection.")
            return df

        # --- Prepare data for ZigZag calculation (potentially resampled) ---
        data_for_zigzag = df.copy() # Default to original (1-min) data
        if self.useAltTF and self.altTF_interval_minutes > 1:
            print(f"{self.strategy_name}: Resampling 1-minute data to {self.altTF_interval_minutes}-minute for ZigZag calculation using k_apis.")
            # Ensure the df passed to convert_minute_data_interval has 'instrument_token' if required by that function
            # The function in myKiteLib.py groups by instrument_token.
            # If df comes from DataPrep.fetch_and_prepare_data, it might not have instrument_token if fetched for single token.
            # Let's assume fetch_and_prepare_data (or the test data in __main__) provides it or adapt.
            # For __main__ test data, we might need to add it if it was single token.
            # However, `convert_minute_data_interval` in `myKiteLib.py` adds it back if missing after grouping.
            # The primary input `df` for `generate_signals` should be the raw 1-min data.
            # `convert_minute_data_interval` expects 'timestamp' and 'instrument_token' columns.
            # `DataPrep.fetch_and_prepare_data` result `ohlcv_data` is indexed by 'date'. 
            # In __main__, we do ohlcv_data.set_index('date').
            # So, before passing to convert_minute_data_interval, ensure it's in expected format.

            # df is 1-min data, indexed by DatetimeIndex (timestamp/date).
            # k_apis.convert_minute_data_interval expects 'timestamp' column and 'instrument_token'.
            # Let's prepare a temporary df for resampling to match expectations if needed.
            df_for_resample_input = df.reset_index() # 'date' or 'index' becomes a column
            if 'date' in df_for_resample_input.columns and 'timestamp' not in df_for_resample_input.columns:
                df_for_resample_input.rename(columns={'date': 'timestamp'}, inplace=True)
            elif 'index' in df_for_resample_input.columns and 'timestamp' not in df_for_resample_input.columns: # if index was unnamed
                 df_for_resample_input.rename(columns={'index': 'timestamp'}, inplace=True)
            
            # Add instrument_token if not present (e.g. for single token data from DataPrep)
            # This relies on self.instrument_token being set, e.g. from config by simulator.
            # For testing, it needs to be available if convert_minute_data_interval strictly needs it.
            # The myKiteLib version seems to handle it if a single group of data is passed.
            # For safety, let's ensure it's there if we have a known token for the df.
            if 'instrument_token' not in df_for_resample_input.columns and hasattr(self, 'instrument_token') and self.instrument_token is not None:
                 df_for_resample_input['instrument_token'] = self.instrument_token
            elif 'instrument_token' not in df_for_resample_input.columns:
                # If instrument_token is critical for convert_minute_data_interval for grouping,
                # and not available, this might be an issue. The sample data in __main__ is single token.
                # Let's add a dummy one for now if missing, assuming single series data.
                # This part is a bit of a workaround if the strategy is only ever given single-token data.
                # The myKiteLib function implies it can handle data for multiple tokens.
                print(f"{self.strategy_name}: Warning - 'instrument_token' column missing for resampling. Adding dummy token 0.")
                df_for_resample_input['instrument_token'] = 0 

            resampled_data = self.k_apis.convert_minute_data_interval(df_for_resample_input, to_interval=self.altTF_interval_minutes)
            
            if resampled_data is not None and not resampled_data.empty:
                # convert_minute_data_interval returns df with 'timestamp' column. Set it as index for _calculate_zigzag_pivots.
                if 'timestamp' in resampled_data.columns:
                    resampled_data['timestamp'] = pd.to_datetime(resampled_data['timestamp'])
                    data_for_zigzag = resampled_data.set_index('timestamp')
                else:
                    print(f"{self.strategy_name}: Warning - Resampled data missing 'timestamp' column. Using 1-minute data for ZigZag.")
            else:
                print(f"{self.strategy_name}: Warning - Resampled data empty after convert_minute_data_interval. Using 1-minute data for ZigZag.")
        else:
            data_for_zigzag = df

        # Calculate all ZigZag pivots once on the (potentially resampled) dataset
        all_zz_pivots_timed = self._calculate_zigzag_pivots(data_for_zigzag) 

        if len(all_zz_pivots_timed) < 5:
            print(f"{self.strategy_name}: Not enough ZigZag pivots found ({len(all_zz_pivots_timed)}) to form XABCD patterns.")
            return df

        # Populate zigzag_price for visualization (maps pivot timestamps to df.index which is 1-min)
        # Pivots in all_zz_pivots_timed have timestamps from data_for_zigzag (e.g., 60-min)
        for pivot in all_zz_pivots_timed:
            # The pivot['timestamp'] is from the resampled data (e.g., 09:15, 10:15 for 60-min)
            # We want to mark this on the original 1-minute df.
            # If df.index is 1-minute, a direct match should work if 60-min interval starts align with a 1-min candle.
            if pivot['timestamp'] in df.index: # df is the 1-minute DataFrame
                df.loc[pivot['timestamp'], 'zigzag_price'] = pivot['price']

        # --- Pattern List for Iteration ---
        # (Function_name_str, Pattern_display_name)
        pattern_checkers = [
            ('_isBat', 'Bat'), ('_isAntiBat', 'Anti Bat'), ('_isAltBat', 'Alt Bat'),
            ('_isButterfly', 'Butterfly'), ('_isAntiButterfly', 'Anti Butterfly'),
            ('_isABCD', 'ABCD'), ('_isGartley', 'Gartley'), ('_isAntiGartley', 'Anti Gartley'),
            ('_isCrab', 'Crab'), ('_isAntiCrab', 'Anti Crab'), ('_isShark', 'Shark'),
            ('_isAntiShark', 'Anti Shark'), ('_is5o', '5-O'), ('_isWolf', 'Wolf Wave'),
            ('_isHnS', 'H&S'), ('_isConTria', 'Cont. Triangle'), ('_isExpTria', 'Exp. Triangle')
        ]
        
        # Reset state variables at the beginning of each call to generate_signals
        self.active_long_trade = None
        self.active_short_trade = None
        self.last_pattern_info = None

        # Rename for clarity
        self.latest_identified_pattern = None

        for i in range(len(df)):
            current_candle = df.iloc[i]
            current_dt = df.index[i]
            current_high = current_candle['high']
            current_low = current_candle['low']
            current_close = current_candle['close']
            
            long_trade_exited_this_candle = False
            short_trade_exited_this_candle = False

            # --- Part 1: ACTIVE TRADE MANAGEMENT - EXIT CHECKS (using TP/SL from *previous* state) ---

            # --- Manage Active Long Trade - EXIT CHECK ---
            if self.active_long_trade and (self.simulation_actual_start_date is None or current_dt.date() >= self.simulation_actual_start_date):
                trade = self.active_long_trade
                # Check for SL/TP Hit using current (i.e., previously calculated) levels
                exit_reason = ""
                if current_high >= trade['current_tp_price']:
                    exit_reason = f"{trade['target_level']} TP Hit at {trade['current_tp_price']:.2f}"
                elif current_low <= trade['current_sl_price']:
                    exit_reason = f"{trade['target_level']} SL Hit at {trade['current_sl_price']:.2f}"
                
                if exit_reason:
                    df.loc[current_dt, 'signal'] = -1 
                    tag = f"{trade['pattern_name']} Long Exit: {exit_reason} (Actual Close: {current_close:.2f}) (Levels from prev. calc)"
                    df.loc[current_dt, 'pattern_tag'] = (df.loc[current_dt, 'pattern_tag'] + " | " + tag 
                                                          if pd.notna(df.loc[current_dt, 'pattern_tag']) else tag)
                    print(f"{current_dt}: {tag}")
                    self.active_long_trade = None
                    long_trade_exited_this_candle = True

            # --- Manage Active Short Trade - EXIT CHECK ---
            if self.active_short_trade and (self.simulation_actual_start_date is None or current_dt.date() >= self.simulation_actual_start_date):
                trade = self.active_short_trade
                exit_reason = ""
                if current_low <= trade['current_tp_price']: # For short, TP is below entry
                    exit_reason = f"{trade['target_level']} TP Hit at {trade['current_tp_price']:.2f}"
                elif current_high >= trade['current_sl_price']: # For short, SL is above entry
                    exit_reason = f"{trade['target_level']} SL Hit at {trade['current_sl_price']:.2f}"

                if exit_reason:
                    df.loc[current_dt, 'signal'] = -1 
                    tag = f"{trade['pattern_name']} Short Exit: {exit_reason} (Actual Close: {current_close:.2f}) (Levels from prev. calc)"
                    df.loc[current_dt, 'pattern_tag'] = (df.loc[current_dt, 'pattern_tag'] + " | " + tag 
                                                          if pd.notna(df.loc[current_dt, 'pattern_tag']) else tag)
                    print(f"{current_dt}: {tag}")
                    self.active_short_trade = None
                    short_trade_exited_this_candle = True
            
            # --- Part 2: Determine Latest ZigZag Pivots (from current candle's data) ---
            pivots_up_to_current_dt = [p for p in all_zz_pivots_timed if p['timestamp'] <= current_dt]
            
            latest_c_pivot_for_calc = None
            latest_d_pivot_for_calc = None
            if len(pivots_up_to_current_dt) >= 2:
                latest_d_pivot_for_calc = pivots_up_to_current_dt[-1]
                latest_c_pivot_for_calc = pivots_up_to_current_dt[-2]

            # --- Part 3: ACTIVE TRADE MANAGEMENT - TP/SL RE-EVALUATION for *NEXT* candle's check ---
            # This happens only if the trade was NOT exited in Part 1.

            # --- Manage Active Long Trade - TP/SL RE-EVALUATION ---
            if self.active_long_trade: # Check if still active
                trade = self.active_long_trade
                if latest_d_pivot_for_calc and latest_c_pivot_for_calc and \
                   latest_d_pivot_for_calc['timestamp'] > trade['last_pivot_d_timestamp_for_calc']:
                    
                    new_tp = self._get_fib_level(latest_c_pivot_for_calc['price'], latest_d_pivot_for_calc['price'], 
                                                 trade['fib_tp_rate'], trade['pattern_mode'] == 1)
                    new_sl = self._get_fib_level(latest_c_pivot_for_calc['price'], latest_d_pivot_for_calc['price'], 
                                                 trade['fib_sl_rate'], trade['pattern_mode'] == 1)

                    if pd.notna(new_tp) and pd.notna(new_sl):
                        # Only update if new TP/SL are different enough to avoid noise, or simply update
                        if abs(trade['current_tp_price'] - new_tp) > 1e-5 or abs(trade['current_sl_price'] - new_sl) > 1e-5:
                            log_msg = (f"{trade['pattern_name']} Long TP/SL re-eval for next check: "
                                       f"Old TP={trade['current_tp_price']:.2f}, Old SL={trade['current_sl_price']:.2f} -> "
                                       f"New TP={new_tp:.2f}, New SL={new_sl:.2f} based on D at {latest_d_pivot_for_calc['timestamp']}")
                            print(f"{current_dt}: {log_msg}")
                            df.loc[current_dt, 'pattern_tag'] = (df.loc[current_dt, 'pattern_tag'] + " | " + log_msg 
                                                                  if pd.notna(df.loc[current_dt, 'pattern_tag']) else log_msg)
                            trade['current_tp_price'] = new_tp
                            trade['current_sl_price'] = new_sl
                            trade['last_pivot_c_price_for_calc'] = latest_c_pivot_for_calc['price']
                            trade['last_pivot_d_price_for_calc'] = latest_d_pivot_for_calc['price']
                            trade['last_pivot_d_timestamp_for_calc'] = latest_d_pivot_for_calc['timestamp']
            
            # --- Manage Active Short Trade - TP/SL RE-EVALUATION ---
            if self.active_short_trade: # Check if still active
                trade = self.active_short_trade
                if latest_d_pivot_for_calc and latest_c_pivot_for_calc and \
                   latest_d_pivot_for_calc['timestamp'] > trade['last_pivot_d_timestamp_for_calc']:

                    new_tp = self._get_fib_level(latest_c_pivot_for_calc['price'], latest_d_pivot_for_calc['price'], 
                                                 trade['fib_tp_rate'], trade['pattern_mode'] == 1) 
                    new_sl = self._get_fib_level(latest_c_pivot_for_calc['price'], latest_d_pivot_for_calc['price'], 
                                                 trade['fib_sl_rate'], trade['pattern_mode'] == 1)
                    
                    if pd.notna(new_tp) and pd.notna(new_sl):
                        if abs(trade['current_tp_price'] - new_tp) > 1e-5 or abs(trade['current_sl_price'] - new_sl) > 1e-5:
                            log_msg = (f"{trade['pattern_name']} Short TP/SL re-eval for next check: "
                                       f"Old TP={trade['current_tp_price']:.2f}, Old SL={trade['current_sl_price']:.2f} -> "
                                       f"New TP={new_tp:.2f}, New SL={new_sl:.2f} based on D at {latest_d_pivot_for_calc['timestamp']}")
                            print(f"{current_dt}: {log_msg}")
                            df.loc[current_dt, 'pattern_tag'] = (df.loc[current_dt, 'pattern_tag'] + " | " + log_msg 
                                                                  if pd.notna(df.loc[current_dt, 'pattern_tag']) else log_msg)
                            trade['current_tp_price'] = new_tp
                            trade['current_sl_price'] = new_sl
                            trade['last_pivot_c_price_for_calc'] = latest_c_pivot_for_calc['price']
                            trade['last_pivot_d_price_for_calc'] = latest_d_pivot_for_calc['price']
                            trade['last_pivot_d_timestamp_for_calc'] = latest_d_pivot_for_calc['timestamp']

            # --- Part 4: IDENTIFY NEW PATTERNS ---
            # This section runs to keep latest_identified_pattern fresh.
            # It uses pivots_up_to_current_dt identified in Part 2.
            
            newly_formed_pattern_details = None 
            
            if len(pivots_up_to_current_dt) >= 5:
                d_pivot = pivots_up_to_current_dt[-1] 
                
                if not self.latest_identified_pattern or \
                   (self.latest_identified_pattern and self.latest_identified_pattern['d_timestamp'] != d_pivot['timestamp']):
                    
                    c_pivot = pivots_up_to_current_dt[-2]
                    b_pivot = pivots_up_to_current_dt[-3]
                    a_pivot = pivots_up_to_current_dt[-4]
                    x_pivot = pivots_up_to_current_dt[-5]
                    
                    x_p, a_p, b_p, c_p, d_p = x_pivot['price'], a_pivot['price'], b_pivot['price'], c_pivot['price'], d_pivot['price']

                    if abs(x_p - a_p) > 1e-6 and abs(a_p - b_p) > 1e-6 and abs(b_p - c_p) > 1e-6:
                        xab = abs(b_p - a_p) / abs(x_p - a_p)
                        abc = abs(b_p - c_p) / abs(a_p - b_p)
                        bcd = abs(c_p - d_p) / abs(b_p - c_p)
                        xad = abs(a_p - d_p) / abs(x_p - a_p)
                        
                        df.loc[current_dt, ['xab_ratio', 'abc_ratio', 'bcd_ratio', 'xad_ratio']] = [xab, abc, bcd, xad]

                        for k_pattern, (pattern_func_str, pattern_name_display_short) in enumerate(pattern_checkers):
                            pattern_checker_method = getattr(self, pattern_func_str)
                            
                            is_bull_pattern = pattern_checker_method(xab, abc, bcd, xad, 1, c_p, d_p)
                            df.loc[current_dt, f'is{self.pattern_names_for_df_columns[k_pattern]}_bull'] = is_bull_pattern
                            
                            is_bear_pattern = pattern_checker_method(xab, abc, bcd, xad, -1, c_p, d_p)
                            df.loc[current_dt, f'is{self.pattern_names_for_df_columns[k_pattern]}_bear'] = is_bear_pattern

                            if newly_formed_pattern_details is None: 
                                if is_bull_pattern:
                                    newly_formed_pattern_details = {
                                        'name': f"Bull {pattern_name_display_short}", 'mode': 1, 
                                        'x_price': x_p, 'a_price': a_p, 'b_price': b_p, 
                                        'c_price': c_p, 'd_price': d_p, 
                                        'x_ts': x_pivot['timestamp'], 'a_ts': a_pivot['timestamp'], 
                                        'b_ts': b_pivot['timestamp'], 'c_ts': c_pivot['timestamp'], 
                                        'd_timestamp': d_pivot['timestamp'],
                                        'ew1_price': self._get_fib_level(c_p, d_p, self.target01_ew_rate, True),
                                        'tp1_price': self._get_fib_level(c_p, d_p, self.target01_tp_rate, True),
                                        'sl1_price': self._get_fib_level(c_p, d_p, self.target01_sl_rate, True),
                                    }
                                elif is_bear_pattern:
                                    newly_formed_pattern_details = {
                                        'name': f"Bear {pattern_name_display_short}", 'mode': -1,
                                        'x_price': x_p, 'a_price': a_p, 'b_price': b_p,
                                        'c_price': c_p, 'd_price': d_p, 
                                        'x_ts': x_pivot['timestamp'], 'a_ts': a_pivot['timestamp'],
                                        'b_ts': b_pivot['timestamp'], 'c_ts': c_pivot['timestamp'],
                                        'd_timestamp': d_pivot['timestamp'],
                                        'ew1_price': self._get_fib_level(c_p, d_p, self.target01_ew_rate, False),
                                        'tp1_price': self._get_fib_level(c_p, d_p, self.target01_tp_rate, False),
                                        'sl1_price': self._get_fib_level(c_p, d_p, self.target01_sl_rate, False),
                                    }
                        
                        if newly_formed_pattern_details:
                            self.latest_identified_pattern = newly_formed_pattern_details
                            tag = f"{self.latest_identified_pattern['name']} Pattern Formed @ D={d_p:.2f} on {d_pivot['timestamp'].strftime('%Y-%m-%d %H:%M')}"
                            df.loc[current_dt, 'pattern_ew1_price'] = self.latest_identified_pattern.get('ew1_price')
                            df.loc[current_dt, 'pattern_tp1_price'] = self.latest_identified_pattern.get('tp1_price')
                            df.loc[current_dt, 'pattern_sl1_price'] = self.latest_identified_pattern.get('sl1_price')
                            
                            if self.simulation_actual_start_date is None or d_pivot['timestamp'].date() >= self.simulation_actual_start_date:
                                df.loc[current_dt, 'pattern_tag'] = (df.loc[current_dt, 'pattern_tag'] + " | " + tag 
                                                                  if pd.notna(df.loc[current_dt, 'pattern_tag']) else tag)
                            print(f"{current_dt}: Potential Pattern Update: {tag}")
                        
                        elif self.latest_identified_pattern and self.latest_identified_pattern['d_timestamp'] != d_pivot['timestamp']:
                            print(f"{current_dt}: New D-point at {d_pivot['timestamp']} did not form a recognized pattern. Clearing latest_identified_pattern.")
                            self.latest_identified_pattern = None
                            df.loc[current_dt, ['pattern_ew1_price', 'pattern_tp1_price', 'pattern_sl1_price']] = np.nan
            
            # --- Part 5: CHECK FOR NEW ENTRIES ---
            if (self.simulation_actual_start_date is None or current_dt.date() >= self.simulation_actual_start_date) and \
               self.latest_identified_pattern and \
               self.latest_identified_pattern['d_timestamp'] <= current_dt: 
                
                pattern_for_entry = self.latest_identified_pattern
                
                if pattern_for_entry['mode'] == 1 and self.active_long_trade is None and not long_trade_exited_this_candle:
                    if pd.notna(pattern_for_entry['ew1_price']) and current_close <= pattern_for_entry['ew1_price']:
                        entry_price_actual = current_close 
                        self.active_long_trade = {
                            'type': 'long', 'pattern_name': pattern_for_entry['name'], 'pattern_mode': 1,
                            'entry_price': entry_price_actual,
                            'initial_tp_price': pattern_for_entry['tp1_price'], 
                            'initial_sl_price': pattern_for_entry['sl1_price'],
                            'current_tp_price': pattern_for_entry['tp1_price'], 
                            'current_sl_price': pattern_for_entry['sl1_price'],
                            'target_level': 'T1',
                            'c_price_entry': pattern_for_entry['c_price'], 'd_price_entry': pattern_for_entry['d_price'],
                            'd_timestamp_entry': pattern_for_entry['d_timestamp'],
                            'last_pivot_c_price_for_calc': pattern_for_entry['c_price'], 
                            'last_pivot_d_price_for_calc': pattern_for_entry['d_price'],
                            'last_pivot_d_timestamp_for_calc': pattern_for_entry['d_timestamp'],
                            'fib_tp_rate': self.target01_tp_rate, 
                            'fib_sl_rate': self.target01_sl_rate
                        }
                        df.loc[current_dt, 'signal'] = 1 
                        entry_log_message = (f"{pattern_for_entry['name']} T1 Long Entry @ {entry_price_actual:.2f} "
                                             f"(Pattern D={pattern_for_entry['d_price']:.2f}, "
                                             f"TP={pattern_for_entry['tp1_price']:.2f}, SL={pattern_for_entry['sl1_price']:.2f})")
                        df.loc[current_dt, 'pattern_tag'] = (df.loc[current_dt, 'pattern_tag'] + " | " + entry_log_message 
                                                              if pd.notna(df.loc[current_dt, 'pattern_tag']) else entry_log_message)
                        print(f"{current_dt}: {entry_log_message}")
                        # long_trade_action_this_candle = True # Not strictly needed here as new entry won't affect same candle exit

                elif pattern_for_entry['mode'] == -1 and self.active_short_trade is None and not short_trade_exited_this_candle:
                     if pd.notna(pattern_for_entry['ew1_price']) and current_close >= pattern_for_entry['ew1_price']:
                        entry_price_actual = current_close 
                        self.active_short_trade = {
                            'type': 'short', 'pattern_name': pattern_for_entry['name'], 'pattern_mode': -1,
                            'entry_price': entry_price_actual,
                            'initial_tp_price': pattern_for_entry['tp1_price'], 
                            'initial_sl_price': pattern_for_entry['sl1_price'],
                            'current_tp_price': pattern_for_entry['tp1_price'], 
                            'current_sl_price': pattern_for_entry['sl1_price'],
                            'target_level': 'T1',
                            'c_price_entry': pattern_for_entry['c_price'], 'd_price_entry': pattern_for_entry['d_price'],
                            'd_timestamp_entry': pattern_for_entry['d_timestamp'],
                            'last_pivot_c_price_for_calc': pattern_for_entry['c_price'],
                            'last_pivot_d_price_for_calc': pattern_for_entry['d_price'],
                            'last_pivot_d_timestamp_for_calc': pattern_for_entry['d_timestamp'],
                            'fib_tp_rate': self.target01_tp_rate, 
                            'fib_sl_rate': self.target01_sl_rate
                        }
                        df.loc[current_dt, 'signal'] = 1 
                        entry_log_message = (f"{pattern_for_entry['name']} T1 Short Entry @ {entry_price_actual:.2f} "
                                             f"(Pattern D={pattern_for_entry['d_price']:.2f}, "
                                             f"TP={pattern_for_entry['tp1_price']:.2f}, SL={pattern_for_entry['sl1_price']:.2f})")
                        df.loc[current_dt, 'pattern_tag'] = (df.loc[current_dt, 'pattern_tag'] + " | " + entry_log_message 
                                                              if pd.notna(df.loc[current_dt, 'pattern_tag']) else entry_log_message)
                        print(f"{current_dt}: {entry_log_message}")
                        # short_trade_action_this_candle = True # Not strictly needed

        print(f"{self.strategy_name}: Signal generation complete.")
        print("Signal counts:\n", df['signal'].value_counts())
        print("Sample pattern tags:\n", df[df['pattern_tag'] != ""]['pattern_tag'].head())
        return df

def calculate_performance_metrics(signals_df: pd.DataFrame, 
                                  initial_capital: float, 
                                  price_column: str = 'close',
                                  **kwargs) -> dict:
    """
    Calculates PnL and other performance metrics from strategy signals.

    Args:
        signals_df: DataFrame from TradingStrategy.generate_signals(), 
                    must include 'signal', 'pattern_tag', and price_column.
                    Index must be a DatetimeIndex.
        initial_capital: The starting capital for the simulation.
        price_column: Column in signals_df to use for entry/exit prices.
        **kwargs: Can include 'annual_rfr' (e.g., 0.02 for 2%).

    Returns:
        A dictionary containing various performance metrics.
    """
    if not isinstance(signals_df.index, pd.DatetimeIndex):
        print("Error in calculate_performance_metrics: signals_df must have a DatetimeIndex.")
        return {}
    if price_column not in signals_df.columns:
        print(f"Error in calculate_performance_metrics: price_column '{price_column}' not found in signals_df.")
        return {}
    if 'signal' not in signals_df.columns:
        print("Error in calculate_performance_metrics: 'signal' column not found in signals_df.")
        return {}
    if 'pattern_tag' not in signals_df.columns:
        print("Error in calculate_performance_metrics: 'pattern_tag' column not found in signals_df.")
        return {}

    active_long_trade = None
    active_short_trade = None
    completed_trades = []

    for current_dt, row in signals_df.iterrows():
        current_price = row[price_column]
        signal_value = row['signal']
        pattern_tag_text = str(row['pattern_tag']).lower() # Ensure string and lowercase

        # Check for exits first based on pattern_tag
        if signal_value == -1: # Potential exit signal
            if "long exit" in pattern_tag_text and active_long_trade:
                pnl = current_price - active_long_trade['entry_price']
                completed_trades.append({
                    'entry_time': active_long_trade['entry_time'],
                    'exit_time': current_dt,
                    'entry_price': active_long_trade['entry_price'],
                    'exit_price': current_price,
                    'trade_type': 'long',
                    'pnl': pnl
                })
                active_long_trade = None
            elif "short exit" in pattern_tag_text and active_short_trade:
                pnl = active_short_trade['entry_price'] - current_price
                completed_trades.append({
                    'entry_time': active_short_trade['entry_time'],
                    'exit_time': current_dt,
                    'entry_price': active_short_trade['entry_price'],
                    'exit_price': current_price,
                    'trade_type': 'short',
                    'pnl': pnl
                })
                active_short_trade = None
        
        # Check for entries
        if signal_value == 1: # Potential entry signal
            if "long entry" in pattern_tag_text and not active_long_trade:
                active_long_trade = {'entry_price': current_price, 'entry_time': current_dt}
            elif "short entry" in pattern_tag_text and not active_short_trade:
                active_short_trade = {'entry_price': current_price, 'entry_time': current_dt}

    metrics = {}
    if not completed_trades:
        metrics['total_trades'] = 0
        metrics['info'] = "No completed trades found to calculate metrics."
        return metrics

    trades_df = pd.DataFrame(completed_trades)
    trades_df.sort_values(by='exit_time', inplace=True) # Ensure chronological order for drawdown

    metrics['total_trades'] = len(trades_df)
    
    winning_trades_df = trades_df[trades_df['pnl'] > 0]
    losing_trades_df = trades_df[trades_df['pnl'] < 0]
    
    metrics['winning_trades'] = len(winning_trades_df)
    metrics['losing_trades'] = len(losing_trades_df)
    
    if metrics['total_trades'] > 0:
        metrics['win_rate_pct'] = (metrics['winning_trades'] / metrics['total_trades']) * 100
    else:
        metrics['win_rate_pct'] = 0

    metrics['total_pnl'] = trades_df['pnl'].sum()
    metrics['gross_profit'] = winning_trades_df['pnl'].sum()
    metrics['gross_loss'] = abs(losing_trades_df['pnl'].sum()) # Sum of absolute losses

    if metrics['total_trades'] > 0:
        metrics['average_pnl_per_trade'] = metrics['total_pnl'] / metrics['total_trades']
    else:
        metrics['average_pnl_per_trade'] = 0
        
    if metrics['winning_trades'] > 0:
        metrics['average_profit_per_winning_trade'] = metrics['gross_profit'] / metrics['winning_trades']
    else:
        metrics['average_profit_per_winning_trade'] = 0

    if metrics['losing_trades'] > 0:
        metrics['average_loss_per_losing_trade'] = metrics['gross_loss'] / metrics['losing_trades'] # gross_loss is positive
    else:
        metrics['average_loss_per_losing_trade'] = 0

    if metrics['gross_loss'] > 0:
        metrics['profit_factor'] = metrics['gross_profit'] / metrics['gross_loss']
    elif metrics['gross_profit'] > 0: # Gross loss is 0 but gross profit > 0
        metrics['profit_factor'] = float('inf') 
    else: # Both are 0
        metrics['profit_factor'] = 0 

    # Maximum Drawdown Calculation
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    trades_df['equity_curve'] = initial_capital + trades_df['cumulative_pnl']
    trades_df['running_max_equity'] = trades_df['equity_curve'].cummax()
    trades_df['drawdown_absolute'] = trades_df['running_max_equity'] - trades_df['equity_curve']
    
    metrics['max_drawdown_absolute'] = trades_df['drawdown_absolute'].max()
    if metrics['max_drawdown_absolute'] is None or pd.isna(metrics['max_drawdown_absolute']):
        metrics['max_drawdown_absolute'] = 0.0
    
    if metrics['max_drawdown_absolute'] > 0:
        idx_max_drawdown = trades_df['drawdown_absolute'].idxmax()
        peak_equity_at_max_drawdown = trades_df.loc[idx_max_drawdown, 'running_max_equity']

        if peak_equity_at_max_drawdown > 0 : 
            metrics['max_drawdown_percentage'] = (metrics['max_drawdown_absolute'] / peak_equity_at_max_drawdown) * 100
        else:
             metrics['max_drawdown_percentage'] = float('inf') 
    else: 
        metrics['max_drawdown_percentage'] = 0.0

    # Sharpe Ratio Calculation (per-trade)
    if len(trades_df) >= 2:
        # Calculate PnL percentage for each trade
        trades_df['pnl_pct'] = trades_df.apply(
            lambda row: ((row['exit_price'] - row['entry_price']) / row['entry_price']) 
                        if row['trade_type'] == 'long' and row['entry_price'] != 0 else \
                        (((row['entry_price'] - row['exit_price']) / row['entry_price'])
                        if row['trade_type'] == 'short' and row['entry_price'] != 0 else 0.0),
            axis=1
        )

        # Calculate holding period in days for each trade
        trades_df['holding_period_days'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / (24 * 60 * 60.0)
        
        # Calculate risk-free rate for each trade based on its holding period
        # annual_rfr is passed as a parameter to the main function, e.g. 0.02 for 2%
        annual_rfr_param = kwargs.get('annual_rfr', 0.02) # Get from kwargs or default
        trades_df['rfr_per_trade'] = (annual_rfr_param / 365.0) * trades_df['holding_period_days']
        
        # Calculate excess return over trade-specific RFR
        trades_df['excess_return_pct'] = trades_df['pnl_pct'] - trades_df['rfr_per_trade']
        
        mean_excess_return_pct = trades_df['excess_return_pct'].mean()
        std_excess_return_pct = trades_df['excess_return_pct'].std()

        if pd.isna(std_excess_return_pct) or std_excess_return_pct < 1e-9: 
            if mean_excess_return_pct > 1e-9: 
                metrics['sharpe_ratio_per_trade'] = float('inf')
            elif mean_excess_return_pct < -1e-9: 
                metrics['sharpe_ratio_per_trade'] = float('-inf')
            else: 
                metrics['sharpe_ratio_per_trade'] = 0.0 
        else:
            metrics['sharpe_ratio_per_trade'] = mean_excess_return_pct / std_excess_return_pct
    else: 
        metrics['sharpe_ratio_per_trade'] = 0.0 

    # For PnL values, let's assume they are per unit.
    # If trade_units are involved, PnL needs to be multiplied by trade_units * price_per_point (if applicable)

    return metrics

if __name__ == '__main__':
    print("--- Testing trading_strategies.py (ZigZag Harmonic Strategy) ---")
    dp = DataPrep()
    if dp.k_apis:
        test_token = 256265 
        sim_start_date = date(2025, 5, 1) 
        sim_end_date = date(2025, 5, 25)   
        warm_up_period_days = 10 # Consistent with config `warm_up_days_for_strategy`
        
        base_data_interval_str = 'minute' 
        print(f"Attempting to fetch 1-MINUTE data for token {test_token} from {sim_start_date} to {sim_end_date} with {warm_up_period_days} warm-up days...")
        
        ohlcv_data_1min = dp.fetch_and_prepare_data(
            instrument_token=test_token,
            start_date_obj=sim_start_date, # This is the intended simulation start
            end_date_obj=sim_end_date,
            interval=base_data_interval_str, 
            warm_up_days=warm_up_period_days
        )

        if ohlcv_data_1min is not None and not ohlcv_data_1min.empty:
            print(f"Fetched data (including warm-up) for {test_token}: {ohlcv_data_1min.shape}")
            # print(ohlcv_data_1min.head())
            
            if 'date' in ohlcv_data_1min.columns:
                ohlcv_data_1min['date'] = pd.to_datetime(ohlcv_data_1min['date'])
                ohlcv_data_1min.set_index('date', inplace=True)
                print("Set 'date' column as DatetimeIndex for the 1-minute data.")
            else:
                print("Error: 'date' column missing in 1-minute data to set as index.")
                # exit() # Keep running to see if calculate_performance_metrics handles it

            strategy_params_from_config = {}
            test_initial_capital = 100000 # Default if not in config
            # In a real scenario, load from trading_config.ini
            # For this test, use defaults or manually set if necessary.
            config = configparser.ConfigParser()
            # Check if config file exists before trying to read
            if os.path.exists('trading_config.ini'):
                config.read('trading_config.ini')
                if 'TRADING_STRATEGY' in config:
                    strategy_params_from_config = dict(config['TRADING_STRATEGY'])
                if 'SIMULATOR_SETTINGS' in config and 'initial_capital' in config['SIMULATOR_SETTINGS']:
                    test_initial_capital = float(config['SIMULATOR_SETTINGS']['initial_capital'])
            else:
                print("Warning: trading_config.ini not found. Using default strategy parameters and initial capital for test.")


            # Pass the original sim_start_date as the actual simulation start date
            strategy_instance = TradingStrategy(
                kite_apis_instance=dp.k_apis, 
                simulation_actual_start_date=sim_start_date, 
                **strategy_params_from_config
            )
            
            print("\nRunning generate_signals with 1-minute data (including warm-up)...")
            visualization_df = strategy_instance.generate_signals(ohlcv_data_1min.copy()) 
            
            print("\n--- DataFrame with Signals (Sample) ---")
            print(visualization_df[['signal', 'pattern_tag', 'close']].head()) # Show relevant columns
            
            print("\n--- Calculating Performance Metrics ---")
            # Use the test_initial_capital defined/loaded above
            performance_results = calculate_performance_metrics(visualization_df, 
                                                                initial_capital=test_initial_capital, 
                                                                price_column='close', # Explicitly pass for clarity
                                                                annual_rfr=0.02) # Pass the desired annual RFR
            
            print("\n--- Performance Metrics Results ---")
            if performance_results:
                for key, value in performance_results.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
            else:
                print("No performance metrics were calculated (e.g., due to data issues or no trades).")
            
            # Existing save to CSV
            output_filename = "zigzag_visualization_data_with_metrics_test.csv" # Changed name slightly
            visualization_df.reset_index().to_csv(output_filename, index=False)
            print(f"\nVisualization data saved to {output_filename}")

        else:
            print(f"Failed to fetch/prepare data. Cannot run full test.")
    else:
        print("DataPrep could not initialize kiteAPIs. Cannot run test.")

    print("\n--- Test Run Complete ---") 