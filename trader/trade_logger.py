import csv
import os
import logging

class TradeLogger:
    """
    Handles logging of completed trades to a CSV file.
    """
    def __init__(self, file_name: str = "trade_log.csv"):
        self.file_path = file_name
        self.logger = logging.getLogger(__name__)
        self._setup_trade_log_file()

    def _setup_trade_log_file(self):
        """Creates the trade log file with headers if it doesn't exist."""
        try:
            if not os.path.exists(self.file_path):
                with open(self.file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    headers = [
                        'timestamp', 'trade_id', 'signal_type', 'pattern_name', 
                        'entry_price', 'exit_price', 'quantity', 'pnl', 
                        'entry_order_id', 'exit_order_id', 'exit_reason', 
                        'tp_price', 'sl_price', 'holding_period_minutes'
                    ]
                    writer.writerow(headers)
                self.logger.info(f"Trade log file created at {self.file_path}")
        except Exception as e:
            self.logger.error(f"Failed to setup trade log file: {e}", exc_info=True)
            raise

    def log_trade(self, trade_data: dict):
        """Appends a single trade record to the CSV file."""
        try:
            with open(self.file_path, 'a', newline='') as f:
                # Ensure all headers are present in the dict to avoid errors
                fieldnames = [
                    'timestamp', 'trade_id', 'signal_type', 'pattern_name', 
                    'entry_price', 'exit_price', 'quantity', 'pnl', 
                    'entry_order_id', 'exit_order_id', 'exit_reason', 
                    'tp_price', 'sl_price', 'holding_period_minutes'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(trade_data)
            self.logger.info(f"Successfully logged trade {trade_data.get('trade_id')}")
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}", exc_info=True) 