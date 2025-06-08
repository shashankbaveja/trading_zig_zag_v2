import logging
import time
from datetime import datetime, timedelta, time as dt_time
import configparser
import pandas as pd

# Refactored local imports
from trader.data_handler import DataHandler
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
    def __init__(self, config_file_path: str, replay_timestamp: datetime = None):
        """
        Initializes the LiveTrader.
        Args:
            config_file_path (str): Path to the configuration file.
            replay_timestamp (datetime, optional): The timestamp to start a replay from. Defaults to None for live mode.
        """
        self.logger = logging.getLogger(__name__)
        self.config_file_path = config_file_path
        self.replay_timestamp = replay_timestamp
        
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
        
        self.strategy_config = self.config['TRADING_STRATEGY']
        self.trader_config = self.config['LIVE_TRADER_SETTINGS']
        self.logger.info("Configuration loaded successfully.")

    def _initialize_components(self):
        """Initializes all the necessary manager and handler components."""
        self.trade_logger = TradeLogger()
        self.order_manager = OrderManager(config=self.trader_config)
        self.session_manager = SessionManager(config=self.trader_config, order_manager=self.order_manager)
        self.position_manager = PositionManager(trade_logger=self.trade_logger)
        self.strategy = ZigZagHarmonicStrategy(**self.strategy_config)
        
        # DataHandler is now initialized without replay information
        self.data_handler = DataHandler(config=self.trader_config, replay_date=None)
        
        self.logger.info("All components initialized.")

    def run_session(self):
        """The main trading loop."""
        self.logger.info("--- Starting Live Trading Session ---")
        
        while self.session_manager.is_session_active:
            loop_start_time = time.time()
            
            try:
                # In replay mode, we don't need real-time session management
                if not self.replay_timestamp:
                    self.session_manager.manage_session()
                    if not self.session_manager.is_trade_allowed:
                        if not self.session_manager.is_session_active:
                            break # Exit loop if session ended
                        time.sleep(self.polling_interval)
                        continue
                
                # Fetch the latest market data, passing the replay timestamp if it exists
                data_dict = self.data_handler.fetch_latest_data(current_timestamp=self.replay_timestamp)
                if not data_dict or data_dict['main_interval_data'].empty:
                    self.logger.warning("No data received from data handler. Skipping tick.")
                    if self.replay_timestamp:
                        self.replay_timestamp += timedelta(seconds=self.polling_interval)
                    else:
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
            
            loop_duration = time.time() - loop_start_time
            if self.replay_timestamp:
                # In replay mode, advance the simulation time
                sleep_time = max(0, self.polling_interval - loop_duration)
                self.replay_timestamp += timedelta(seconds=(loop_duration + sleep_time))

                # If the replay time is past the market close, jump to the next day's open
                if self.replay_timestamp.time() > dt_time(15, 30):
                    self.logger.info(f"Replay timestamp {self.replay_timestamp} is past 15:30. Jumping to next trading day.")
                    next_day = self.replay_timestamp.date() + timedelta(days=1)
                    # Set to 9:15 AM for the next session
                    self.replay_timestamp = self.replay_timestamp.replace(
                        year=next_day.year, month=next_day.month, day=next_day.day,
                        hour=9, minute=15, second=0, microsecond=0
                    )
                
                self.logger.info(f"Replay time advanced to: {self.replay_timestamp}")

                # The original polling interval logic is now handled by advancing the timestamp.
                # We keep a small, fixed sleep to control replay speed without affecting time calculations.
                time.sleep(0.2)
            else:
                sleep_time = max(0, self.polling_interval - loop_duration)
                time.sleep(sleep_time)
            
        self.logger.info(f"--- Live Trading Session Ended. Final Daily PnL: {self.daily_pnl:.2f} ---")

    def _handle_active_trade(self, latest_candle: pd.Series):
        """Checks for an exit signal on the currently active trade."""
        exit_reason = self.position_manager.check_for_exit(latest_candle)
        if exit_reason:
            self.logger.info(f"Exit signal received: {exit_reason}")
            
            active_trade = self.position_manager.active_trade
            
            # In replay mode, we simulate the order placement
            if self.replay_timestamp:
                exit_order_id = f"REPLAY_EXIT_{int(self.replay_timestamp.timestamp())}"
                self.logger.info(f"SIMULATING EXIT ORDER: {exit_order_id}")
            else:
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
        signal = self.strategy.check_for_signal(
            data_dict=data_dict,
            latest_price=latest_price,
            replay_timestamp=self.replay_timestamp
        )
        if signal:
            self.logger.info(f"New executable trade signal from strategy: {signal['pattern_name']}")
            self._execute_new_trade(signal, latest_price)

    def _execute_new_trade(self, signal: dict, entry_price: float):
        """Executes and starts tracking a new trade."""
        current_time = self.replay_timestamp if self.replay_timestamp else datetime.now()
        trade_id = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{signal['direction']}"
        
        # In replay mode, we simulate the order placement
        if self.replay_timestamp:
            entry_order_id = f"REPLAY_ENTRY_{int(self.replay_timestamp.timestamp())}"
            self.logger.info(f"SIMULATING ENTRY ORDER: {entry_order_id}")
        else:
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