import pandas as pd
import numpy as np
from collections import deque
import logging

from strategy.indicators import calculate_zigzag_pivots
from strategy.patterns import ALL_PATTERNS

class ZigZagHarmonicStrategy:
    """
    Implements the ZigZag Harmonic trading strategy in a stateless manner.
    Its primary role is to identify new trade entry opportunities.
    """
    def __init__(self, **kwargs):
        """
        Initializes the strategy with parameters.
        Args:
            **kwargs: Strategy parameters, typically from a config file.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("ZigZagHarmonicStrategy Initializing...")
        
        # Parameters for TP/SL calculation
        self.target01_ew_rate = float(kwargs.get('target01_ew_rate', 0.236))
        self.target01_tp_rate = float(kwargs.get('target01_tp_rate', 0.618))
        self.target01_sl_rate = float(kwargs.get('target01_sl_rate', -0.236))
        
        # Parameters for alternate timeframe analysis
        self.useAltTF = str(kwargs.get('usealttf', 'true')).lower() == 'true'
        self.altTF_interval_minutes = int(kwargs.get('alttf_interval_minutes', 10))
        
        # State tracking for pattern detection
        self.last_processed_d_pivot_ts = None
        self.recent_pivots = deque(maxlen=10)
        
        self.logger.info(f"Strategy initialized with TP rate: {self.target01_tp_rate}, SL rate: {self.target01_sl_rate}")

    def check_for_signal(self, data_dict: dict) -> dict | None:
        """
        Analyzes the latest market data to find a new trading signal.

        Args:
            data_dict (dict): A dictionary containing 'main_interval_data' and 'one_minute_data'.

        Returns:
            A dictionary representing the trade signal if a new pattern is found, otherwise None.
            Signal Format: {
                'pattern_name': str, 'direction': 'LONG'/'SHORT', 'entry_price': float,
                'target_price': float, 'stop_loss_price': float, 'pattern_info': dict
            }
        """
        # Determine which data to use for ZigZag calculation
        if self.useAltTF and self.altTF_interval_minutes > 1:
            # Note: For live trading, resampling needs to be handled carefully.
            # This implementation assumes data_dict provides correctly resampled altTF data if needed.
            data_for_zigzag = data_dict.get('alt_tf_data', data_dict['main_interval_data'])
            self.logger.debug(f"Using alternate timeframe data ({len(data_for_zigzag)} rows) for ZigZag.")
        else:
            data_for_zigzag = data_dict['main_interval_data']
            self.logger.debug(f"Using main interval data ({len(data_for_zigzag)} rows) for ZigZag.")

        if data_for_zigzag.empty or len(data_for_zigzag) < 10:
            self.logger.warning("Data for ZigZag is too short. Cannot generate signals.")
            return None

        # 1. Calculate ZigZag pivots
        all_pivots = calculate_zigzag_pivots(data_for_zigzag, self.recent_pivots)
        if len(all_pivots) < 5:
            return None # Not enough pivots to form a pattern

        # 2. Check the latest pivot point
        d_pivot = all_pivots[-1]
        
        # Avoid re-processing the same pattern on subsequent ticks
        if self.last_processed_d_pivot_ts and self.last_processed_d_pivot_ts == d_pivot['timestamp']:
            return None
        
        self.logger.info(f"New pivot D detected at {d_pivot['timestamp']} ({d_pivot['price']:.2f}). Checking for patterns.")
        
        # 3. Get the 5 points (X, A, B, C, D) for pattern analysis
        c_pivot, b_pivot, a_pivot, x_pivot = all_pivots[-2], all_pivots[-3], all_pivots[-4], all_pivots[-5]
        x_p, a_p, b_p, c_p, d_p = x_pivot['price'], a_pivot['price'], b_pivot['price'], c_pivot['price'], d_pivot['price']

        # 4. Calculate Fibonacci ratios
        try:
            xab = abs(b_p - a_p) / abs(x_p - a_p)
            abc = abs(b_p - c_p) / abs(a_p - b_p)
            bcd = abs(c_p - d_p) / abs(b_p - c_p)
            xad = abs(a_p - d_p) / abs(x_p - a_p)
        except ZeroDivisionError:
            self.logger.warning("Division by zero in ratio calculation. Skipping pivot.")
            self.last_processed_d_pivot_ts = d_pivot['timestamp'] # Mark as processed to avoid loops
            return None

        # 5. Iterate through all patterns to find a match
        for pattern in ALL_PATTERNS:
            # Check for bullish pattern
            if pattern['func'](xab, abc, bcd, xad, 1, c_p, d_p):
                self.last_processed_d_pivot_ts = d_pivot['timestamp']
                return self._create_signal(f"Bull {pattern['name']}", 'LONG', c_p, d_p, d_pivot)

            # Check for bearish pattern
            if pattern['func'](xab, abc, bcd, xad, -1, c_p, d_p):
                self.last_processed_d_pivot_ts = d_pivot['timestamp']
                return self._create_signal(f"Bear {pattern['name']}", 'SHORT', c_p, d_p, d_pivot)
        
        # If no pattern was found for this new pivot, mark it as processed
        self.last_processed_d_pivot_ts = d_pivot['timestamp']
        self.logger.info(f"No valid harmonic pattern found for D-pivot at {d_pivot['timestamp']}.")
        return None

    def _get_fib_level(self, c_price, d_price, rate, is_bullish):
        """Calculates Fibonacci level."""
        fib_range = abs(d_price - c_price)
        if is_bullish:
            return d_price + (fib_range * rate)
        else:
            return d_price - (fib_range * rate)

    def _create_signal(self, pattern_name: str, direction: str, c_price: float, d_price: float, d_pivot: dict) -> dict:
        """Constructs the signal dictionary."""
        is_bullish = (direction == 'LONG')
        
        entry_window_price = self._get_fib_level(c_price, d_price, self.target01_ew_rate, is_bullish)
        target_price = self._get_fib_level(c_price, d_price, self.target01_tp_rate, is_bullish)
        stop_loss_price = self._get_fib_level(c_price, d_price, self.target01_sl_rate, is_bullish)
        
        signal = {
            'pattern_name': pattern_name,
            'direction': direction,
            'entry_window_price': entry_window_price,
            'target_price': target_price,
            'stop_loss_price': stop_loss_price,
            'pattern_info': {
                'd_price': d_price,
                'd_timestamp': d_pivot['timestamp']
            }
        }
        
        self.logger.info(f"SIGNAL CREATED: {signal}")
        return signal 