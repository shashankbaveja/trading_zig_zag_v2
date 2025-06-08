import pandas as pd
import numpy as np
from collections import deque
import logging
from datetime import datetime

from strategy.indicators import calculate_zigzag_pivots
from strategy.patterns import ALL_PATTERNS
from strategy.strategy_logger import StrategyLogger

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
        
        # Enhanced logging for debugging and review
        self.debug_logger = StrategyLogger()

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
        self.pending_signal = None
        
        self.logger.info(f"Strategy initialized with TP rate: {self.target01_tp_rate}, SL rate: {self.target01_sl_rate}")

    def check_for_signal(self, data_dict: dict, latest_price: float, replay_timestamp: datetime | None = None) -> dict | None:
        """
        Statefully analyzes market data to find and validate trading signals.
        1. Identifies new harmonic patterns and stores them as a 'pending_signal'.
        2. If a pending_signal exists, it validates that the pattern is still intact.
        3. If the pattern is valid, it checks if the latest_price is within the entry window.
        4. If all conditions are met, it returns an executable signal.

        Args:
            data_dict (dict): Contains 'main_interval_data'.
            latest_price (float): The current market price to check against the entry window.
            replay_timestamp (datetime | None): The current timestamp in replay mode.

        Returns:
            A dictionary representing the trade signal if all conditions are met, otherwise None.
        """
        run_timestamp = datetime.now()
        data_for_pivots = data_dict['main_interval_data']

        if data_for_pivots.empty or len(data_for_pivots) < 10:
            return None

        # Determine resampling interval for pivot calculation
        resample_interval = self.altTF_interval_minutes if self.useAltTF else 1
        
        # 1. Calculate ZigZag pivots
        all_pivots = calculate_zigzag_pivots(
            data=data_for_pivots, recent_pivots_deque=self.recent_pivots,
            resample_interval_minutes=resample_interval
        )
        
        if len(all_pivots) < 5:
            if self.pending_signal:
                self.logger.warning("Pivots dropped below 5, invalidating pending signal.")
                self.pending_signal = None
            return None

        # 2. Check for a new pattern formation
        d_pivot = all_pivots[-1]
        if not self.last_processed_d_pivot_ts or self.last_processed_d_pivot_ts != d_pivot['timestamp']:
            self.logger.info(f"New pivot D detected at {d_pivot['timestamp']} ({d_pivot['price']:.2f}). Checking for new patterns.")
            self.last_processed_d_pivot_ts = d_pivot['timestamp']
            self.pending_signal = self._evaluate_new_pattern(all_pivots, run_timestamp, replay_timestamp)

        # 3. If a pending signal exists, check if it's ready to be triggered
        if self.pending_signal:
            # First, ensure the pending signal is still valid based on the latest pivots
            if d_pivot['timestamp'] != self.pending_signal['pattern_info']['d_timestamp']:
                self.logger.info("A newer D-pivot has formed, invalidating the previous pending signal.")
                self.pending_signal = None # Invalidate signal if a newer D pivot appears
                return None

            # Check if the price is within the entry window
            if self._is_price_in_entry_window(latest_price, self.pending_signal):
                self.logger.info(f"Price {latest_price:.2f} is within entry window for pending signal {self.pending_signal['pattern_name']}.")
                executable_signal = self.pending_signal
                self.pending_signal = None # Consume the signal
                self.debug_logger.log_signal(run_timestamp, executable_signal, replay_timestamp)
                return executable_signal
            else:
                self.logger.debug(f"Pending signal '{self.pending_signal['pattern_name']}' exists, but price {latest_price:.2f} is outside entry window.")
        
        return None

    def _evaluate_new_pattern(self, all_pivots: list, run_timestamp: datetime, replay_timestamp: datetime | None) -> dict | None:
        """
        Takes the latest pivots and checks if a new harmonic pattern has formed.
        Returns a pending signal dictionary if a pattern is found, else None.
        """
        last_5_pivots = all_pivots[-5:]
        d_pivot = last_5_pivots[-1]
        
        self.debug_logger.log_pivots(run_timestamp, all_pivots[-15:], replay_timestamp)

        x_pivot, a_pivot, b_pivot, c_pivot = last_5_pivots[0], last_5_pivots[1], last_5_pivots[2], last_5_pivots[3]
        x_p, a_p, b_p, c_p, d_p = x_pivot['price'], a_pivot['price'], b_pivot['price'], c_pivot['price'], d_pivot['price']

        try:
            ratios = {
                'xab': abs(b_p - a_p) / abs(x_p - a_p) if (x_p - a_p) != 0 else 0,
                'abc': abs(b_p - c_p) / abs(a_p - b_p) if (a_p - b_p) != 0 else 0,
                'bcd': abs(c_p - d_p) / abs(b_p - c_p) if (b_p - c_p) != 0 else 0,
                'xad': abs(a_p - d_p) / abs(x_p - a_p) if (x_p - a_p) != 0 else 0
            }
        except ZeroDivisionError:
            self.logger.warning("Division by zero in ratio calculation. Cannot evaluate pattern.")
            return None

        for pattern in ALL_PATTERNS:
            for direction_mode, direction_name, pattern_prefix in [(1, 'Bullish', 'Bull'), (-1, 'Bearish', 'Bear')]:
                is_match = pattern['func'](ratios['xab'], ratios['abc'], ratios['bcd'], ratios['xad'], direction_mode, c_p, d_p)
                self.debug_logger.log_pattern_check(run_timestamp, pattern['name'], direction_name, ratios, is_match, replay_timestamp)
                if is_match:
                    full_pattern_name = f"{pattern_prefix} {pattern['name']}"
                    pending_signal = self._create_signal_shell(full_pattern_name, direction_name.upper(), c_p, d_p, d_pivot)
                    self.logger.info(f"New PENDING SIGNAL identified: {full_pattern_name}")
                    return pending_signal
        
        self.logger.info(f"No valid harmonic pattern found for D-pivot at {d_pivot['timestamp']}.")
        return None
        
    def _is_price_in_entry_window(self, price: float, signal: dict) -> bool:
        """Checks if the price is valid for entry based on the signal."""
        entry_window = signal['entry_window_price']
        if signal['direction'] == 'LONG':
            return price <= entry_window
        else: # SHORT
            return price >= entry_window

    def _get_fib_level(self, c_price, d_price, rate, is_bullish):
        """Calculates Fibonacci level."""
        fib_range = abs(d_price - c_price)
        if is_bullish:
            return d_price + (fib_range * rate)
        else:
            return d_price - (fib_range * rate)

    def _create_signal_shell(self, pattern_name: str, direction: str, c_price: float, d_price: float, d_pivot: dict) -> dict:
        """Constructs the signal dictionary shell, without logging it as final."""
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
        
        # Do not log here, only log when the signal becomes executable
        return signal 