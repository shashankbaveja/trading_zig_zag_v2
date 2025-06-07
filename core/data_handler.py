import pandas as pd
from datetime import date, timedelta, datetime
from myKiteLib import kiteAPIs

class DataHandler:
    """
    Handles fetching and preparing historical market data for strategies and live trading.
    """
    def __init__(self, kite_apis: kiteAPIs = None, config: dict = None, replay_date: date = None):
        """
        Initializes the DataHandler.
        Args:
            kite_apis (kiteAPIs, optional): An initialized kiteAPIs instance. 
            config (dict, optional): Live trader settings for token info.
            replay_date (date, optional): If provided, the handler will fetch data for this date.
        """
        self.replay_date = replay_date
        try:
            self.k_apis = kite_apis if kite_apis else kiteAPIs()
            self.signal_token = int(config.get('signal_token', 256265)) if config else 256265
            self.data_lookback_days = int(config.get('data_lookback_days', 5)) if config else 5
            
            if self.replay_date:
                print(f"DataHandler initialized in REPLAY MODE for date: {self.replay_date}")
            else:
                print("DataHandler initialized in LIVE MODE.")
        except Exception as e:
            print(f"DataHandler Error: Could not initialize kiteAPIs: {e}")
            self.k_apis = None

    def fetch_historical_data(self, instrument_token: int, start_date_obj: date, 
                               end_date_obj: date, interval_minutes: int = 1) -> pd.DataFrame:
        """
        Fetches a block of historical data for backtesting or initial warm-up.
        """
        start_date_str = start_date_obj.strftime('%Y-%m-%d')
        end_date_str = end_date_obj.strftime('%Y-%m-%d')
        
        raw_data = self.k_apis.extract_data_from_db(
            from_date=start_date_str, to_date=end_date_str,
            interval='minute', instrument_token=instrument_token
        )
        if raw_data is None or raw_data.empty:
            return pd.DataFrame()

        cleaned_data = self._clean_raw_data(raw_data)
        
        if interval_minutes > 1:
            return self._resample_data(cleaned_data, interval_minutes, instrument_token)
        
        return cleaned_data

    def fetch_latest_data(self) -> dict | None:
        """
        Fetches the most recent data needed for a single live trading tick.
        In replay mode, 'latest' refers to the data for the specified replay_date.
        """
        if not self.k_apis:
            print("DataHandler Error: kiteAPIs not initialized.")
            return None

        # Use the replay_date if provided, otherwise use today's date for live trading
        end_date = self.replay_date if self.replay_date else date.today()
        start_date = end_date - timedelta(days=self.data_lookback_days)
        
        # This uses the direct API call in myKiteLib, which should be faster for live data
        raw_data = self.k_apis.getHistoricalData(
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d'),
            tokens=[self.signal_token],
            interval='minute'
        )

        if raw_data is None or raw_data.empty:
            print(f"DataHandler: No recent data found for token {self.signal_token}.")
            return None

        # The getHistoricalData from myKiteLib already returns a clean DataFrame
        # but we ensure it's processed consistently.
        # It might be better to unify the data source (e.g., always from DB and have a separate script to populate DB).
        # For now, we assume getHistoricalData returns a DF compatible with _clean_raw_data
        
        if 'date' not in raw_data.columns and 'timestamp' in raw_data.columns:
            raw_data.rename(columns={'timestamp':'date'}, inplace=True)
            
        cleaned_data = self._clean_raw_data(raw_data)
        
        if cleaned_data.empty:
            return None
            
        # Ignore the latest, potentially incomplete candle for strategy calculation
        if len(cleaned_data) > 1:
            cleaned_data = cleaned_data.iloc[:-1]

        return {
            'main_interval_data': cleaned_data.copy(),
            'one_minute_data': cleaned_data.copy()
            # In the future, alt_tf_data could be resampled here if needed
        }

    def _clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and standardizes the raw data fetched from the source."""
        if 'timestamp' in df.columns and 'date' not in df.columns:
            df.rename(columns={'timestamp': 'date'}, inplace=True)
        
        if 'date' not in df.columns:
            print("DataHandler Critical Error: 'date' column not found.")
            return pd.DataFrame()
            
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            print(f"DataHandler Critical Error: Failed to convert 'date' to datetime: {e}.")
            return pd.DataFrame()
        
        cols_to_numeric = ['open', 'high', 'low', 'close', 'volume']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.sort_values(by='date', inplace=True)
        df.drop_duplicates(subset=['date'], keep='first', inplace=True)
        df.dropna(subset=['date', 'open', 'high', 'low', 'close'], inplace=True)
        df.set_index('date', inplace=True)
        
        return df

    def _resample_data(self, one_min_df: pd.DataFrame, interval_mins: int, token: int) -> pd.DataFrame:
        """Resamples 1-minute data to the specified interval."""
        if interval_mins <= 1:
            return one_min_df.copy()

        print(f"DataHandler: Resampling 1-minute data to {interval_mins}-minute interval...")
        if not hasattr(self.k_apis, 'convert_minute_data_interval'):
            print("DataHandler Error: kiteAPIs object missing 'convert_minute_data_interval' method.")
            return pd.DataFrame()

        try:
            df_for_resample = one_min_df.reset_index().rename(columns={'date': 'timestamp'})
            if 'instrument_token' not in df_for_resample.columns:
                df_for_resample['instrument_token'] = token
            
            resampled_df = self.k_apis.convert_minute_data_interval(df_for_resample, to_interval=interval_mins)
            
            if resampled_df is None or resampled_df.empty:
                print(f"DataHandler Error: Resampling to '{interval_mins}' mins resulted in empty data.")
                return pd.DataFrame()
            
            return self._clean_raw_data(resampled_df)

        except Exception as e:
            print(f"DataHandler Error during resampling: {e}")
            return pd.DataFrame() 