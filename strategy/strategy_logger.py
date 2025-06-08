import os
import logging
from datetime import datetime
import pandas as pd

class StrategyLogger:
    """
    Handles logging of detailed strategy calculations to separate CSV files for debugging and analysis.
    """
    def __init__(self, log_dir: str = 'logs/strategy_debug'):
        """
        Initializes the logger, creates the log directory, and sets up the CSV files with headers.
        """
        self.logger = logging.getLogger(__name__)
        self.log_dir = log_dir
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(self.log_dir, exist_ok=True)

        self.pivots_log_path = os.path.join(self.log_dir, f'pivots_{run_id}.csv')
        self.patterns_log_path = os.path.join(self.log_dir, f'patterns_{run_id}.csv')
        self.signals_log_path = os.path.join(self.log_dir, f'signals_{run_id}.csv')
        
        self._init_csv_files()
        self.logger.info(f"StrategyLogger initialized. Logging to directory: {self.log_dir}")

    def _init_csv_files(self):
        """Create CSV files with headers if they don't exist."""
        try:
            if not os.path.exists(self.pivots_log_path):
                pd.DataFrame(columns=[
                    'run_timestamp', 'replay_timestamp', 'pivot_label', 
                    'pivot_timestamp', 'price', 'type'
                ]).to_csv(self.pivots_log_path, index=False)
                
            if not os.path.exists(self.patterns_log_path):
                pd.DataFrame(columns=[
                    'run_timestamp', 'replay_timestamp', 'pattern_name', 'direction', 
                    'xab', 'abc', 'bcd', 'xad', 'is_match'
                ]).to_csv(self.patterns_log_path, index=False)

            if not os.path.exists(self.signals_log_path):
                pd.DataFrame(columns=[
                    'run_timestamp', 'replay_timestamp', 'pattern_name', 'direction', 
                    'entry_window_price', 'target_price', 'stop_loss_price', 
                    'd_price', 'd_timestamp'
                ]).to_csv(self.signals_log_path, index=False)
        except Exception as e:
            self.logger.error(f"StrategyLogger: Failed to initialize CSV files: {e}")

    def log_pivots(self, run_timestamp, pivots: list, replay_timestamp: datetime | None = None):
        """Logs the list of pivots used for a pattern check."""
        if not pivots:
            return
        
        try:
            labels = ['D', 'C', 'B', 'A', 'X']
            entries = []
            for i, pivot in enumerate(reversed(pivots)):
                label = labels[i] if i < len(labels) else f"Prev-{i - len(labels) + 1}"
                
                entry = {
                    'run_timestamp': run_timestamp,
                    'replay_timestamp': replay_timestamp,
                    'pivot_timestamp': pivot['timestamp'],
                    'price': pivot['price'],
                    'type': pivot.get('type', 'N/A'),
                    'pivot_label': label
                }
                entries.append(entry)
            
            log_df = pd.DataFrame(reversed(entries))
            log_df.to_csv(self.pivots_log_path, mode='a', header=False, index=False)
        except Exception as e:
            self.logger.error(f"StrategyLogger Error: Could not log pivots - {e}")

    def log_pattern_check(self, run_timestamp, pattern_name, direction, ratios, result, replay_timestamp: datetime | None = None):
        """Logs the result of a single pattern check."""
        try:
            log_entry = pd.DataFrame([{
                'run_timestamp': run_timestamp,
                'replay_timestamp': replay_timestamp,
                'pattern_name': pattern_name,
                'direction': direction,
                'xab': ratios.get('xab'),
                'abc': ratios.get('abc'),
                'bcd': ratios.get('bcd'),
                'xad': ratios.get('xad'),
                'is_match': result
            }])
            log_entry.to_csv(self.patterns_log_path, mode='a', header=False, index=False)
        except Exception as e:
            self.logger.error(f"StrategyLogger Error: Could not log pattern check - {e}")

    def log_signal(self, run_timestamp, signal: dict, replay_timestamp: datetime | None = None):
        """Logs the details of a generated signal."""
        try:
            log_entry = pd.DataFrame([{
                'run_timestamp': run_timestamp,
                'replay_timestamp': replay_timestamp,
                'pattern_name': signal['pattern_name'],
                'direction': signal['direction'],
                'entry_window_price': signal['entry_window_price'],
                'target_price': signal['target_price'],
                'stop_loss_price': signal['stop_loss_price'],
                'd_price': signal['pattern_info']['d_price'],
                'd_timestamp': signal['pattern_info']['d_timestamp']
            }])
            log_entry.to_csv(self.signals_log_path, mode='a', header=False, index=False)
        except Exception as e:
            self.logger.error(f"StrategyLogger Error: Could not log signal - {e}") 