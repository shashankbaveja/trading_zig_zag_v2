import pandas as pd
import os
import logging
from datetime import datetime

class TradeLogger:
    """Handles logging of all completed trades to a CSV file."""
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.file_path = os.path.join(self.log_dir, "trade_log.csv")
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(self.log_dir, exist_ok=True)
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Creates the log file with a header if it doesn't exist."""
        if not os.path.exists(self.file_path):
            header = [
                'timestamp', 'trade_id', 'signal_type', 'pattern_name',
                'entry_price', 'exit_price', 'quantity', 'pnl',
                'entry_order_id', 'exit_order_id', 'exit_reason',
                'tp_price', 'sl_price', 'holding_period_minutes'
            ]
            try:
                pd.DataFrame(columns=header).to_csv(self.file_path, index=False)
                self.logger.info(f"Trade log created at {self.file_path}")
            except Exception as e:
                self.logger.error(f"Failed to create trade log: {e}")

    def log_trade(self, trade_data: dict):
        """Appends a completed trade to the CSV log."""
        try:
            trade_df = pd.DataFrame([trade_data])
            trade_df.to_csv(self.file_path, mode='a', header=False, index=False)
            self.logger.info(f"Logged trade {trade_data.get('trade_id')} to {self.file_path}")
        except Exception as e:
            self.logger.error(f"Failed to log trade: {e}") 