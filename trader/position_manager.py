import logging
from datetime import datetime
import pandas as pd

class PositionManager:
    """
    Manages the state of an active trade, including its exit conditions.
    This class is responsible for what happens *after* a trade is entered.
    """
    def __init__(self, trade_logger):
        """
        Initializes the PositionManager.
        Args:
            trade_logger: An instance of a CSV logger for recording trades.
        """
        self.logger = logging.getLogger(__name__)
        self.trade_logger = trade_logger
        
        # State variables
        self.active_trade = None
        
        self.logger.info("PositionManager initialized.")

    @property
    def is_trade_active(self) -> bool:
        """Returns True if a trade is currently active."""
        return self.active_trade is not None

    def start_new_trade(self, trade_id: str, direction: str, entry_price: float, 
                        tp_price: float, sl_price: float, quantity: int, 
                        pattern_name: str, entry_order_id: str):
        """
        Begins tracking a new active trade.
        """
        if self.is_trade_active:
            self.logger.warning("Attempted to start a new trade, but one is already active.")
            return

        self.active_trade = {
            'trade_id': trade_id,
            'direction': direction,
            'pattern_name': pattern_name,
            'entry_timestamp': datetime.now(),
            'entry_price': entry_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'quantity': quantity,
            'entry_order_id': entry_order_id,
            'status': 'ACTIVE'
        }
        self.logger.info(f"PositionManager started tracking new {direction} trade: {trade_id}")
        self.logger.info(f"  -> Entry: {entry_price:.2f}, TP: {tp_price:.2f}, SL: {sl_price:.2f}")

    def check_for_exit(self, latest_candle: dict) -> str | None:
        """
        Checks if the active trade should be exited based on the latest candle.
        
        Args:
            latest_candle (dict): A dictionary representing the most recent candle (with 'high', 'low').
            
        Returns:
            A string with the exit reason if an exit is triggered, otherwise None.
        """
        if not self.is_trade_active:
            return None

        trade = self.active_trade
        direction = trade['direction']
        tp = trade['tp_price']
        sl = trade['sl_price']
        
        high_price = latest_candle['high']
        low_price = latest_candle['low']
        
        exit_reason = None
        if direction == 'LONG':
            if not pd.isna(tp) and high_price >= tp:
                exit_reason = f"TP_HIT({high_price:.2f} >= {tp:.2f})"
            elif not pd.isna(sl) and low_price <= sl:
                exit_reason = f"SL_HIT({low_price:.2f} <= {sl:.2f})"
        
        elif direction == 'SHORT':
            if not pd.isna(tp) and low_price <= tp:
                exit_reason = f"TP_HIT({low_price:.2f} <= {tp:.2f})"
            elif not pd.isna(sl) and high_price >= sl:
                exit_reason = f"SL_HIT({high_price:.2f} >= {sl:.2f})"
        
        return exit_reason

    def close_trade(self, exit_price: float, exit_reason: str, exit_order_id: str) -> float:
        """
        Closes the active trade, calculates PnL, logs it, and resets state.
        
        Returns:
            The calculated PnL for the closed trade.
        """
        if not self.is_trade_active:
            self.logger.warning("close_trade called, but no active trade to close.")
            return 0.0

        trade = self.active_trade
        
        # Calculate PnL
        if trade['direction'] == 'LONG':
            pnl_per_unit = exit_price - trade['entry_price']
        else: # SHORT
            pnl_per_unit = trade['entry_price'] - exit_price
        
        total_pnl = pnl_per_unit * trade['quantity']
        
        # Calculate holding period
        holding_period_minutes = (datetime.now() - trade['entry_timestamp']).total_seconds() / 60
        
        # Log the completed trade
        trade_log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'trade_id': trade['trade_id'],
            'signal_type': trade['direction'],
            'pattern_name': trade['pattern_name'],
            'entry_price': trade['entry_price'],
            'exit_price': exit_price,
            'quantity': trade['quantity'],
            'pnl': total_pnl,
            'entry_order_id': trade['entry_order_id'],
            'exit_order_id': exit_order_id,
            'exit_reason': exit_reason,
            'tp_price': trade['tp_price'],
            'sl_price': trade['sl_price'],
            'holding_period_minutes': round(holding_period_minutes, 2)
        }
        self.trade_logger.log_trade(trade_log_data)
        
        self.logger.info(f"Closing trade {trade['trade_id']}. Reason: {exit_reason}. PnL: {total_pnl:.2f}")
        
        # Reset state
        self.active_trade = None
        
        return total_pnl 