import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from ta.volatility import DonchianChannel
from myKiteLib import kiteAPIs # Assuming myKiteLib.py is in the same directory or accessible in PYTHONPATH
from datetime import date, datetime # For type hinting and date conversions
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

    def fetch_and_prepare_data(self, instrument_token: int, start_date_obj: date, end_date_obj: date, interval: str = 'minute') -> pd.DataFrame:
        """
        Fetches 1-minute historical data from the database via myKiteLib, 
        then resamples it to the specified 'interval' if needed, and prepares it.

        Args:
            instrument_token: The instrument token to fetch data for.
            start_date_obj: The start date for data fetching (datetime.date object).
            end_date_obj: The end date for data fetching (datetime.date object).
            interval: The target candle interval (e.g., 'minute', '5minute', '15minute').

        Returns:
            A pandas DataFrame with prepared OHLCV data at the target interval,
            sorted by date, with a 'date' column (datetime), and numeric OHLCV columns.
            Returns an empty DataFrame if fetching or preparation fails.
        """
        if not self.k_apis:
            print("DataPrep Error: kiteAPIs not initialized. Cannot fetch data.")
            return pd.DataFrame()

        start_date_str = start_date_obj.strftime('%Y-%m-%d')
        end_date_str = end_date_obj.strftime('%Y-%m-%d')

        # Always fetch 1-minute data from the database
        db_fetch_interval = 'minute' # Or whatever string your DB function expects for 1-minute
        print(f"DataPrep: Fetching 1-minute data for token {instrument_token} from {start_date_str} to {end_date_str}...")
        
        historical_df_minute = self.k_apis.extract_data_from_db(
            from_date=start_date_str,
            to_date=end_date_str,
            interval=db_fetch_interval, # Always fetch minute data
            instrument_token=instrument_token
        )

        if historical_df_minute is None or historical_df_minute.empty:
            print(f"DataPrep: No 1-minute data fetched for token {instrument_token} from {start_date_str} to {end_date_str}.")
            return pd.DataFrame()

        print(f"DataPrep: Successfully fetched {len(historical_df_minute)} rows of 1-minute data. Preparing...")

        # Basic preparation of minute data
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
        # Do not reset_index here if convert_minute_data_interval expects the DatetimeIndex for resampling

        # Resample if target interval is different from 1-minute
        target_interval_minutes = self._parse_interval_string(interval)

        if target_interval_minutes > 1:
            print(f"DataPrep: Resampling 1-minute data to {target_interval_minutes}-minute interval...")
            if not hasattr(self.k_apis, 'convert_minute_data_interval'):
                print("DataPrep Error: kiteAPIs object does not have method 'convert_minute_data_interval'. Cannot resample.")
                # Decide if to return minute data or empty. For now, returning minute if resampling fails here.
                print("DataPrep Warning: Proceeding with 1-minute data as resampling function is missing.")
                final_df = historical_df_minute
            else:
                try:
                    # Ensure 'date' is the index for resampling if convert_minute_data_interval expects it
                    # It's safer if convert_minute_data_interval itself handles setting the index if needed.
                    # Assuming convert_minute_data_interval can take a DataFrame and a to_interval integer.
                    final_df = self.k_apis.convert_minute_data_interval(historical_df_minute.copy(), to_interval=target_interval_minutes)
                    if final_df is None or final_df.empty:
                        print(f"DataPrep Error: Resampling to {target_interval_minutes}-minute resulted in empty data. Check resampling logic.")
                        return pd.DataFrame()
                    print(f"DataPrep: Resampled to {len(final_df)} rows at {target_interval_minutes}-minute interval.")
                except Exception as e:
                    print(f"DataPrep Error during resampling to {target_interval_minutes}-minute: {e}")
                    print("DataPrep Warning: Proceeding with 1-minute data due to resampling error.")
                    final_df = historical_df_minute # Fallback to minute data
        else:
            final_df = historical_df_minute # Use 1-minute data directly

        # Final preparation on the (potentially resampled) DataFrame
        final_df.reset_index(drop=True, inplace=True) # Reset index after any resampling
        
        required_cols = ['date', 'open', 'high', 'low', 'close'] 
        final_df.dropna(subset=required_cols, inplace=True)

        if not all(col in final_df.columns for col in required_cols):
            print(f"DataPrep Error: Missing one or more required columns {required_cols} after final preparation.")
            return pd.DataFrame()

        print(f"DataPrep: Data preparation complete for interval '{interval}'. Final shape: {final_df.shape}")
        return final_df

    def calculate_statistics(self, data: pd.DataFrame, donchian_length: int = -1, rsi_period: int = -1, ma_short_period: int = -1, ma_long_period: int = -1) -> pd.DataFrame:
        """
        Calculates a standard set of technical indicators and adds them to the DataFrame.
        Currently calculates Donchian Channels, RSI, and Moving Averages.

        Args:
            data: DataFrame with OHLCV data.
            donchian_length: The lookback period for the Donchian Channel. 
                             If -1, tries to load from config or uses a hardcoded fallback.
            rsi_period: The lookback period for the RSI indicator.
                        If -1, tries to load from config or uses a hardcoded fallback.
            ma_short_period: The lookback period for the short-term Moving Average.
                             If -1, tries to load from config or uses a hardcoded fallback.
            ma_long_period: The lookback period for the long-term Moving Average.
                             If -1, tries to load from config or uses a hardcoded fallback.

        Returns:
            DataFrame with added indicator columns.
        """
        config = configparser.ConfigParser()
        # Attempt to read the config file. Adjust path if necessary.
        # This assumes trading_config.ini is in the same directory as this script
        # or in a known location. For robustness, the path might need to be passed
        # or determined more dynamically in a larger application.
        config_path = os.path.join(os.path.dirname(__file__), 'trading_config.ini')
        try:
            if not os.path.exists(config_path):
                # Try one level up if not found (common in some project structures)
                config_path_alt = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trading_config.ini')
                if os.path.exists(config_path_alt):
                    config_path = config_path_alt
                else:
                    print(f"DataPrep Warning: trading_config.ini not found at {config_path} or parent. Using hardcoded defaults for DataPrep.")
            
            if os.path.exists(config_path):
                config.read(config_path)
                # print(f"DataPrep: Successfully read {config_path}") # For debugging
            else:
                # If config file doesn't exist after checks, config object will be empty. get/getint will use fallbacks.
                pass # Allow to proceed and use fallbacks
        except Exception as e:
            print(f"DataPrep Warning: Error reading trading_config.ini: {e}. Using hardcoded defaults for DataPrep.")

        if donchian_length == -1: # If no specific length provided, use config or hardcoded default
            resolved_donchian_length = config.getint('DATA_PREP_DEFAULTS', 'default_donchian_length_for_dp', fallback=20)
            # print(f"DataPrep: Using Donchian length from config/fallback: {resolved_donchian_length}") # For debugging
        else:
            resolved_donchian_length = donchian_length

        if not isinstance(resolved_donchian_length, int) or resolved_donchian_length <= 0:
            raise ValueError("Donchian Channel length must be a positive integer for statistics calculation.")

        df = data.copy()
        print(f"DataPrep: Calculating statistics - Donchian Channels (length {resolved_donchian_length})...")

        # Calculate Donchian Channels using ta library
        donchian_indicator = DonchianChannel(
            high=df['high'],
            low=df['low'],
            close=df['close'], # Not strictly used by ta.DonchianChannel but often passed
            window=resolved_donchian_length,
            fillna=False 
        )
        df['don_upper'] = donchian_indicator.donchian_channel_hband()
        df['don_lower'] = donchian_indicator.donchian_channel_lband()
        df['don_basis'] = (df['don_upper'] + df['don_lower']) / 2 # Calculate basis explicitly

        # Shift indicator values for use in strategies (previous candle's value for current decision)
        df['don_upper_prev'] = df['don_upper'].shift(1)
        df['don_lower_prev'] = df['don_lower'].shift(1)
        df['don_basis_prev'] = df['don_basis'].shift(1)
        
        # --- ADD NEW INDICATORS START ---

        # Resolve RSI Period
        if rsi_period == -1:
            resolved_rsi_period = config.getint('DATA_PREP_DEFAULTS', 'default_rsi_period_for_dp', fallback=14)
        else:
            resolved_rsi_period = rsi_period
        if not isinstance(resolved_rsi_period, int) or resolved_rsi_period <= 0:
            print("DataPrep Warning: RSI period must be a positive integer. Using fallback 14.")
            resolved_rsi_period = 14
            
        # Resolve MA Short Period
        if ma_short_period == -1:
            resolved_ma_short_period = config.getint('DATA_PREP_DEFAULTS', 'default_ma_short_period_for_dp', fallback=5)
        else:
            resolved_ma_short_period = ma_short_period
        if not isinstance(resolved_ma_short_period, int) or resolved_ma_short_period <= 0:
            print("DataPrep Warning: MA Short period must be a positive integer. Using fallback 5.")
            resolved_ma_short_period = 5

        # Resolve MA Long Period
        if ma_long_period == -1:
            resolved_ma_long_period = config.getint('DATA_PREP_DEFAULTS', 'default_ma_long_period_for_dp', fallback=200)
        else:
            resolved_ma_long_period = ma_long_period
        if not isinstance(resolved_ma_long_period, int) or resolved_ma_long_period <= 0:
            print("DataPrep Warning: MA Long period must be a positive integer. Using fallback 200.")
            resolved_ma_long_period = 200

        # Calculate RSI using ta library if period is valid
        if resolved_rsi_period > 0 and 'close' in df.columns and not df['close'].empty:
            from ta.momentum import RSIIndicator # Import here
            print(f"DataPrep: - RSI (period {resolved_rsi_period})")
            rsi_indicator = RSIIndicator(close=df['close'], window=resolved_rsi_period, fillna=False)
            df['rsi'] = rsi_indicator.rsi()
            df['rsi_prev'] = df['rsi'].shift(1)
        else:
            print(f"DataPrep: Skipping RSI calculation (period: {resolved_rsi_period}, data valid: {'close' in df.columns and not df['close'].empty})")

        # Calculate Short-term Moving Average using ta library if period is valid
        if resolved_ma_short_period > 0 and 'close' in df.columns and not df['close'].empty:
            from ta.trend import SMAIndicator # Import here
            print(f"DataPrep: - Short MA (period {resolved_ma_short_period})")
            sma_short_indicator = SMAIndicator(close=df['close'], window=resolved_ma_short_period, fillna=False)
            df['ma_short'] = sma_short_indicator.sma_indicator()
            df['ma_short_prev'] = df['ma_short'].shift(1)
        else:
            print(f"DataPrep: Skipping Short MA calculation (period: {resolved_ma_short_period}, data valid: {'close' in df.columns and not df['close'].empty})")

        # Calculate Long-term Moving Average using ta library if period is valid
        if resolved_ma_long_period > 0 and 'close' in df.columns and not df['close'].empty:
            from ta.trend import SMAIndicator # Import here
            print(f"DataPrep: - Long MA (period {resolved_ma_long_period})")
            sma_long_indicator = SMAIndicator(close=df['close'], window=resolved_ma_long_period, fillna=False)
            df['ma_long'] = sma_long_indicator.sma_indicator()
            df['ma_long_prev'] = df['ma_long'].shift(1)
        else:
            print(f"DataPrep: Skipping Long MA calculation (period: {resolved_ma_long_period}, data valid: {'close' in df.columns and not df['close'].empty})")
        
        # --- ADD NEW INDICATORS END ---

        # TODO: Add other common indicators here as needed for other strategies (RSI, BBands, ADX, etc.)
        # Example for RSI (uncomment and adapt if needed later):
        # from ta.momentum import RSIIndicator
        # rsi_indicator = RSIIndicator(close=df['close'], window=14, fillna=False)
        # df['rsi'] = rsi_indicator.rsi()
        # df['rsi_prev'] = df['rsi'].shift(1)

        print(f"DataPrep: Statistics calculation complete. DataFrame shape: {df.shape}")
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

class DonchianBreakoutStrategy(BaseStrategy):
    """
    Implements a Donchian Channel Breakout strategy.
    Long Entry: Close crosses above the previous upper Donchian band.
    Long Exit: Based on one of two options:
                1. Close crosses below the previous lower Donchian band.
                2. Close crosses below the previous middle Donchian band (basis).
    """
    def __init__(self, length: int = 20, exit_option: int = 1):
        """
        Initializes the DonchianBreakoutStrategy.

        Args:
            length: The lookback period for the Donchian Channel (default 20).
            exit_option: The exit logic to use (default 1).
                         1: Exit on crossunder of the lower band.
                         2: Exit on crossunder of the basis line (midpoint).
        """
        if not isinstance(length, int) or length <= 0:
            raise ValueError("Donchian Channel length must be a positive integer.")
        if exit_option not in [1, 2]:
            raise ValueError("Exit option must be 1 (lower band) or 2 (basis line).")
        
        self.length = length
        self.exit_option = exit_option
        self.strategy_name = f"DonchianBO_L{self.length}_Exit{self.exit_option}"
        print(f"Initialized {self.strategy_name}")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates BUY (1), SELL/EXIT (-1), or HOLD (0) signals.
        Assumes required indicator columns (e.g., 'don_upper_prev') are already in the input DataFrame.
        """
        # Define required columns for this strategy
        required_indicator_cols = ['don_upper_prev', 'don_lower_prev', 'don_basis_prev']
        base_ohlcv_cols = ['high', 'low', 'close', 'open', 'date']
        all_required_cols = base_ohlcv_cols + required_indicator_cols

        if not all(col in data.columns for col in all_required_cols):
            missing_cols = [col for col in all_required_cols if col not in data.columns]
            raise ValueError(f"Input DataFrame for {self.strategy_name} must contain required columns. Missing: {missing_cols}")
        
        df = data.copy() # Work on a copy

        # Initialize signal column (0: HOLD, 1: BUY, -1: SELL/EXIT)
        df['signal'] = 0 

        # --- Entry Condition ---
        # Long Entry: current close crosses over previous upper band
        # crossover(close, upper[1]) in Pine means: close > upper[1] AND close[1] <= upper[2] (or upper[1] of previous bar)
        # A simpler way, and common for breakout systems: close > upper[1]
        # The Pine Script crossover: (df['close'] > df['don_upper_prev']) & (df['close'].shift(1) <= df['don_upper_prev'].shift(1))
        
        # For BUY signal:
        # Price crosses above previous Donchian Upper band
        long_entry_condition = (df['close'] > df['don_upper_prev']) & \
                               (df['close'].shift(1) <= df['don_upper_prev'].shift(1)) # Condition for strict crossover
        df.loc[long_entry_condition, 'signal'] = 1

        # --- Exit Conditions ---
        # Define exit conditions based on the chosen option
        if self.exit_option == 1:
            # Exit if current close crosses under previous lower band
            long_exit_condition = (df['close'] < df['don_lower_prev']) & \
                                  (df['close'].shift(1) >= df['don_lower_prev'].shift(1))
        elif self.exit_option == 2:
            # Exit if current close crosses under previous basis line
            long_exit_condition = (df['close'] < df['don_basis_prev']) & \
                                  (df['close'].shift(1) >= df['don_basis_prev'].shift(1))
        else: # Should not happen due to __init__ validation
            long_exit_condition = pd.Series([False] * len(df), index=df.index)

        df.loc[long_exit_condition, 'signal'] = -1
        
        # Ensure that a SELL signal only occurs if a BUY signal could have been active.
        # This logic is simplified: the simulator will manage actual position state.
        # A BUY signal (1) indicates an intention to go long.
        # A SELL signal (-1) indicates an intention to exit a long position.
        # If a BUY and SELL signal occur on the same bar due to logic, priority might be needed
        # or simulator handles it. For now, let them be distinct.
        # Example: if long_entry_condition is true AND long_exit_condition is true on the same bar,
        # 'signal' would be -1 as it's the last assignment. This might be okay if exit has priority.

        # Fill NaN signals with 0 (HOLD) - especially for the initial period where indicators are not yet calculated
        df['signal'] = df['signal'].fillna(0)
        df['signal'] = df['signal'].astype(int)

        print(f"{self.strategy_name}: Signals generated.DataFrame shape: {df.shape}")
        print(df['signal'].value_counts())
        
        return df

class MovingAverageRSILong(BaseStrategy):
    """
    Implements a strategy based on RSI(2) and Moving Averages for LONG signals.
    Long Entry: Close > Long MA AND Close < Short MA AND RSI(2) < Oversold Threshold.
    """
    def __init__(self, rsi_period: int = 2, rsi_oversold_threshold: int = 10, 
                 ma_short_period: int = 5, ma_long_period: int = 200,
                 signal_offset_period: int = 1): # Default to 1 (no offset)
        if not all(isinstance(p, int) and p > 0 for p in [rsi_period, ma_short_period, ma_long_period]):
            raise ValueError("RSI and MA periods must be positive integers.")
        if not (isinstance(rsi_oversold_threshold, int) and 0 < rsi_oversold_threshold < 100):
            raise ValueError("RSI oversold threshold must be an integer between 0 and 100.")
        if not (isinstance(signal_offset_period, int) and signal_offset_period >= 1):
            raise ValueError("Signal offset period must be an integer greater than or equal to 1.")

        self.rsi_period = rsi_period
        self.rsi_oversold_threshold = rsi_oversold_threshold
        self.ma_short_period = ma_short_period
        self.ma_long_period = ma_long_period
        self.signal_offset_period = signal_offset_period
        self.strategy_name = f"MARSI_Long_R{self.rsi_period}_T{self.rsi_oversold_threshold}_S{self.ma_short_period}_L{self.ma_long_period}_O{self.signal_offset_period}"
        print(f"Initialized {self.strategy_name}")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = ['close', 'rsi_prev', 'ma_short_prev', 'ma_long_prev']
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            raise ValueError(f"Input DataFrame for {self.strategy_name} is missing columns: {missing}")

        df = data.copy()
        df['signal'] = 0
        df['raw_signal'] = False # Helper column for raw condition

        # Calculate raw entry condition
        raw_long_entry_condition = (
            (df['close'] > df['ma_long_prev']) &
            (df['close'] < df['ma_short_prev']) &
            (df['rsi_prev'] < self.rsi_oversold_threshold)
        )
        df.loc[raw_long_entry_condition, 'raw_signal'] = True

        # Calculate consecutive raw signals
        df['consecutive_raw_signals'] = 0
        consecutive_count = 0
        for i in range(len(df)):
            if df.loc[i, 'raw_signal']:
                consecutive_count += 1
            else:
                consecutive_count = 0
            df.loc[i, 'consecutive_raw_signals'] = consecutive_count
        
        # Generate final signal based on offset
        final_long_entry_condition = (df['raw_signal']) & (df['consecutive_raw_signals'] == self.signal_offset_period)
        df.loc[final_long_entry_condition, 'signal'] = 1

        df['signal'] = df['signal'].fillna(0).astype(int)
        # Drop helper columns if not needed for debugging output
        # df.drop(columns=['raw_signal', 'consecutive_raw_signals'], inplace=True)
        print(f"{self.strategy_name}: Signals generated. DataFrame shape: {df.shape}")
        print(df['signal'].value_counts())
        return df

class MovingAverageRSIShort(BaseStrategy):
    """
    Implements a strategy based on RSI(2) and Moving Averages for SHORT signals.
    Short Entry: Close < Long MA AND Close > Short MA AND RSI(2) > Overbought Threshold.
    (Signal is 1, option_type=PE in config will make it a short trade)
    """
    def __init__(self, rsi_period: int = 2, rsi_overbought_threshold: int = 90, 
                 ma_short_period: int = 5, ma_long_period: int = 200,
                 signal_offset_period: int = 1): # Default to 1 (no offset)
        if not all(isinstance(p, int) and p > 0 for p in [rsi_period, ma_short_period, ma_long_period]):
            raise ValueError("RSI and MA periods must be positive integers.")
        if not (isinstance(rsi_overbought_threshold, int) and 0 < rsi_overbought_threshold < 100):
            raise ValueError("RSI overbought threshold must be an integer between 0 and 100.")
        if not (isinstance(signal_offset_period, int) and signal_offset_period >= 1):
            raise ValueError("Signal offset period must be an integer greater than or equal to 1.")

        self.rsi_period = rsi_period
        self.rsi_overbought_threshold = rsi_overbought_threshold
        self.ma_short_period = ma_short_period
        self.ma_long_period = ma_long_period
        self.signal_offset_period = signal_offset_period
        self.strategy_name = f"MARSI_Short_R{self.rsi_period}_T{self.rsi_overbought_threshold}_S{self.ma_short_period}_L{self.ma_long_period}_O{self.signal_offset_period}"
        print(f"Initialized {self.strategy_name}")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = ['close', 'rsi_prev', 'ma_short_prev', 'ma_long_prev']
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            raise ValueError(f"Input DataFrame for {self.strategy_name} is missing columns: {missing}")

        df = data.copy()
        df['signal'] = 0
        df['raw_signal'] = False # Helper column for raw condition

        # Calculate raw entry condition
        # For shorting, we generate a BUY signal (1) which the simulator will use to buy a PE option.
        raw_short_entry_condition = (
            (df['close'] < df['ma_long_prev']) &
            (df['close'] > df['ma_short_prev']) &
            (df['rsi_prev'] > self.rsi_overbought_threshold)
        )
        df.loc[raw_short_entry_condition, 'raw_signal'] = True

        # Calculate consecutive raw signals
        df['consecutive_raw_signals'] = 0
        consecutive_count = 0
        for i in range(len(df)):
            if df.loc[i, 'raw_signal']:
                consecutive_count += 1
            else:
                consecutive_count = 0
            df.loc[i, 'consecutive_raw_signals'] = consecutive_count

        # Generate final signal based on offset
        final_short_entry_condition = (df['raw_signal']) & (df['consecutive_raw_signals'] == self.signal_offset_period)
        df.loc[final_short_entry_condition, 'signal'] = 1 # Signal 1, simulator buys PE for short

        df['signal'] = df['signal'].fillna(0).astype(int)
        # Drop helper columns if not needed for debugging output
        # df.drop(columns=['raw_signal', 'consecutive_raw_signals'], inplace=True)
        print(f"{self.strategy_name}: Signals generated. DataFrame shape: {df.shape}")
        print(df['signal'].value_counts())
        return df

if __name__ == '__main__':
    print("--- Testing trading_strategies.py ---")
    
    # Test DataPrep
    print("\n--- Testing DataPrep ---")
    dp = DataPrep()
    if dp.k_apis: # Proceed only if kiteAPIs initialized
        # NIFTY 50 Index token = 256265
        # Using a very short period for quick testing. Replace with actual dates for real use.
        test_token = 256265 
        # Ensure you have data for this period in your DB for this token
        # For this example, let's use a date far in the past to avoid issues if DB is empty recently
        # and hope there's *some* data. User must ensure DB has data for this test.
        start_test_date = date(2025, 5, 1) 
        end_test_date = date(2025, 5, 16) 
        
        print(f"Attempting to fetch data for token {test_token} from {start_test_date} to {end_test_date}...")
        sample_ohlcv_data = dp.fetch_and_prepare_data(
            instrument_token=test_token,
            start_date_obj=start_test_date,
            end_date_obj=end_test_date,
            interval='minute'
        )

        if not sample_ohlcv_data.empty:
            print("\nFetched and prepared sample data:")
            print(sample_ohlcv_data.head())
            print(f"Data shape: {sample_ohlcv_data.shape}")

            # Calculate all statistics
            print("\n--- Calculating All Statistics via DataPrep ---")
            # Use the length defined by the strategy instance for consistency
            donchian_strat = DonchianBreakoutStrategy(length=20, exit_option=1) # Create instance to get its length
            
            data_with_all_stats = dp.calculate_statistics(
                sample_ohlcv_data.copy(), 
                donchian_length=donchian_strat.length
            )

            if data_with_all_stats.empty:
                print("Failed to calculate statistics. Exiting test.")
                exit()

            print("\nData with all stats (first 5 rows of Donchian columns):")
            donchian_cols_to_show = [col for col in ['date', 'don_upper', 'don_lower', 'don_basis', 'don_upper_prev', 'don_lower_prev', 'don_basis_prev'] if col in data_with_all_stats.columns]
            print(data_with_all_stats[donchian_cols_to_show].head())

            # Test DonchianBreakoutStrategy
            print("\n--- Testing DonchianBreakoutStrategy (using pre-calculated stats) ---")
            # The strategy instance donchian_strat is already created above
            data_with_signals = donchian_strat.generate_signals(data_with_all_stats.copy()) # Pass a copy
            
            print("\nData with Donchian signals (last 10 rows):")
            # Display relevant columns for signal checking
            cols_to_show = ['date', 'open', 'high', 'low', 'close', 'don_upper_prev', 'don_lower_prev', 'don_basis_prev', 'signal']
            display_cols = [col for col in cols_to_show if col in data_with_signals.columns]
            print(data_with_signals[display_cols].tail(10))
            
            print("\nSignal counts:")
            print(data_with_signals['signal'].value_counts())

            # Test with exit_option = 2
            donchian_strat_2 = DonchianBreakoutStrategy(length=20, exit_option=2)
            # No need to recalculate stats if length is the same. 
            # If strategy had different length, we would call calculate_statistics again with that length.
            # For this test, we assume data_with_all_stats is suitable (calculated with length 20).
            data_with_signals_2 = donchian_strat_2.generate_signals(data_with_all_stats.copy())
            print("\nSignal counts (Exit Option 2):")
            print(data_with_signals_2['signal'].value_counts())

        else:
            print(f"Failed to fetch sample data for token {test_token}. Cannot run strategy test.")
            print("Please ensure your database has data for the test token and period,")
            print("and that myKiteLib.py and database credentials in security.txt are correctly configured.")
    else:
        print("DataPrep could not initialize kiteAPIs. Cannot run tests.")

    print("\n--- trading_strategies.py testing complete ---") 