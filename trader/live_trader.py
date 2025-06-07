import logging
import time
from datetime import datetime
import configparser
import pandas as pd

# Refactored local imports
from core.data_handler import DataHandler
from strategy.zigzag_harmonic import ZigZagHarmonicStrategy
from trader.session_manager import SessionManager
from trader.order_manager import OrderManager
from trader.position_manager import PositionManager
from trader.trade_logger import TradeLogger

class LiveTrader:
    """
    The central orchestrator for the live trading bot.
    Connects all the modular components to run the trading session.
    """
    def __init__(self, config_file_path: str, replay_date: str = None):
        """
        Initializes the LiveTrader.
        Args:
            config_file_path (str): Path to the configuration file.
            replay_date (str, optional): The date to run in replay mode (YYYY-MM-DD). Defaults to None for live mode.
        """
        self.logger = logging.getLogger(__name__)
        self.config_file_path = config_file_path
        
        self.replay_date_obj = datetime.strptime(replay_date, '%Y-%m-%d').date() if replay_date else None
        
        self._load_config()
        self._initialize_components()

        self.polling_interval = self.trader_config.getint('polling_interval_seconds', 15)
        self.daily_pnl = 0.0
        self.logger.info("LiveTrader orchestration engine initialized.")

    def _load_config(self):
        """Loads settings from the configuration file."""
        self.config = configparser.ConfigParser()
        if not self.config.read(self.config_file_path):
            raise FileNotFoundError(f"Config file not found: {self.config_file_path}")
        
        self.strategy_config = dict(self.config['TRADING_STRATEGY'])
        self.trader_config = dict(self.config['LIVE_TRADER_SETTINGS'])
        self.logger.info("Configuration loaded successfully.")

    def _initialize_components(self):
        """Initializes all the necessary manager and handler components."""
        self.trade_logger = TradeLogger()
        self.order_manager = OrderManager(config=self.trader_config)
        self.session_manager = SessionManager(config=self.trader_config, order_manager=self.order_manager)
        self.position_manager = PositionManager(trade_logger=self.trade_logger)
        self.strategy = ZigZagHarmonicStrategy(**self.strategy_config)
        
        # The DataHandler is now initialized with the replay date if provided
        self.data_handler = DataHandler(config=self.trader_config, replay_date=self.replay_date_obj)
        
        self.logger.info("All components initialized.")

    def run_session(self):
        """The main trading loop."""
        self.logger.info("--- Starting Live Trading Session ---")
        
        while self.session_manager.is_session_active:
            loop_start_time = time.time()
            
            try:
                self.session_manager.manage_session()

                # If in replay mode, we can ignore real-time session checks
                is_trading_time = self.session_manager.is_trade_allowed
                if self.replay_date_obj:
                    is_trading_time = True # Always allow trading during replay

                if not is_trading_time:
                    if not self.session_manager.is_session_active:
                        break # Exit loop if session ended
                    time.sleep(self.polling_interval)
                    continue

                # Fetch the latest market data
                data_dict = self.data_handler.fetch_latest_data()
                if not data_dict or data_dict['main_interval_data'].empty:
                    self.logger.warning("No data received from data handler. Skipping tick.")
                    time.sleep(self.polling_interval)
                    continue

                latest_candle = data_dict['main_interval_data'].iloc[-1]
                latest_price = latest_candle['close']
                self.logger.debug(f"Latest Price: {latest_price:.2f} at {latest_candle.name}")

                # --- Core Trading Logic ---
                if self.position_manager.is_trade_active:
                    self._handle_active_trade(latest_candle)
                else:
                    self._check_for_new_trade(data_dict, latest_price)

            except KeyboardInterrupt:
                self.logger.warning("Keyboard interrupt received. Shutting down.")
                break
            except Exception as e:
                self.logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)

            # Maintain polling interval
            loop_duration = time.time() - loop_start_time
            sleep_time = max(0, self.polling_interval - loop_duration)
            time.sleep(sleep_time)
            
        self.logger.info(f"--- Live Trading Session Ended. Final Daily PnL: {self.daily_pnl:.2f} ---")

    def _handle_active_trade(self, latest_candle: pd.Series):
        """Checks for an exit signal on the currently active trade."""
        exit_reason = self.position_manager.check_for_exit(latest_candle)
        if exit_reason:
            self.logger.info(f"Exit signal received: {exit_reason}")
            
            active_trade = self.position_manager.active_trade
            exit_order_id = self.order_manager.place_exit_order(
                direction=active_trade['direction'],
                reason=exit_reason
            )
            
            if exit_order_id:
                # For simplicity, we use the candle's close as exit price.
                # In a real scenario, you might confirm with order execution details.
                pnl = self.position_manager.close_trade(
                    exit_price=latest_candle['close'],
                    exit_reason=exit_reason,
                    exit_order_id=exit_order_id
                )
                self.daily_pnl += pnl
            else:
                self.logger.error("Failed to place exit order. Trade remains open.")

    def _check_for_new_trade(self, data_dict: dict, latest_price: float):
        """Checks for a new entry signal from the strategy."""
        signal = self.strategy.check_for_signal(data_dict)
        if signal:
            self.logger.info(f"New trade signal from strategy: {signal['pattern_name']}")
            
            # Check if current price is within the entry window
            if self._is_price_in_entry_window(latest_price, signal):
                self._execute_new_trade(signal, latest_price)
            else:
                self.logger.info("Signal found, but current price is outside the entry window. No action taken.")

    def _is_price_in_entry_window(self, price: float, signal: dict) -> bool:
        """Checks if the price is valid for entry based on the signal."""
        entry_window = signal['entry_window_price']
        if signal['direction'] == 'LONG':
            return price <= entry_window
        else: # SHORT
            return price >= entry_window

    def _execute_new_trade(self, signal: dict, entry_price: float):
        """Executes and starts tracking a new trade."""
        trade_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal['direction']}"
        
        entry_order_id = self.order_manager.place_entry_order(
            direction=signal['direction'],
            pattern_name=signal['pattern_name']
        )
        
        if entry_order_id:
            self.logger.info(f"Successfully placed entry order {entry_order_id} for trade {trade_id}.")
            self.position_manager.start_new_trade(
                trade_id=trade_id,
                direction=signal['direction'],
                entry_price=entry_price, # Using latest price as assumed entry
                tp_price=signal['target_price'],
                sl_price=signal['stop_loss_price'],
                quantity=self.order_manager.quantity,
                pattern_name=signal['pattern_name'],
                entry_order_id=entry_order_id
            )
        else:
            self.logger.error(f"Failed to execute new trade for signal: {signal['pattern_name']}") 