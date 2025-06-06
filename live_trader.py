import logging
import time
import csv
import os
from datetime import datetime, date, timedelta, time as dt_time
import configparser
import pandas as pd
import numpy as np
import sys

# Import existing components
from myKiteLib import OrderPlacement, kiteAPIs
from trading_strategies import TradingStrategy, calculate_performance_metrics

# --- Constants ---
LOG_FILE_NAME = "live_trader.log"
TRADE_LOG_FILE_NAME = "trade_log.csv"
NIFTY_DATA_LOOKBACK_DAYS = 5  # Default, will be overridden by config
POLLING_INTERVAL_SECONDS = 15  # Default, will be overridden by config
MAX_ENTRY_ORDER_RETRIES = 2
ENTRY_ORDER_RETRY_DELAY_SECONDS = 3

class LiveTrader:
    def __init__(self, config_file_path: str):
        self.config_file_path = config_file_path
        self.config = configparser.ConfigParser()
        
        if not os.path.exists(self.config_file_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file_path}")
        
        self.config.read(self.config_file_path)
        self._setup_logging()
        self.logger.info("--- LiveTrader (ZigZag Harmonic) Initializing ---")

        # Set up a dedicated logger for live trader actions
        self.action_logger = logging.getLogger('live_trader_actions')
        self.action_logger.setLevel(logging.INFO)
        action_handler = logging.FileHandler('logs/live_trader_actions.log', mode='w')
        action_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        self.action_logger.addHandler(action_handler)

        # Initialize core components
        try:
            self.order_manager = OrderPlacement()
            self.logger.info("OrderPlacement initialized. Attempting to initialize trading session...")
            self.order_manager.init_trading()  # This handles token validation
            self.logger.info("Kite trading session initialized successfully.")

            self.k_apis = kiteAPIs()
            self.logger.info("kiteAPIs initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize core components: {e}", exc_info=True)
            raise SystemExit(f"Core component initialization failed: {e}")

        # Load strategy and settings
        self._load_strategy_from_config()
        self._load_trader_settings()
        self._setup_trade_logging()

        # State variables for trading
        self.active_trades = {}  # Dict to store multiple active trades if needed
        self.is_any_trade_active = False
        self.trading_session_enabled = True
        self.system_healthy = False
        self.trade_execution_allowed = False
        self.last_signal_timestamp = None
        self.latest_identified_pattern = None

        # Performance tracking
        self.daily_pnl = 0.0
        self.trade_count_today = 0

        self.logger.info(f"LiveTrader Initialized. Polling: {self.polling_interval_seconds}s")
        self.logger.info(f"Trading Window: {self.trading_start_time_str}-{self.trading_end_time_str}")
        self.logger.info(f"Signal Token: {self.signal_token}, Trading Token: {self.trading_token}")

    def _setup_logging(self):
        """Setup logging to both file and console"""
        # Remove existing log file to start fresh
        if os.path.exists(LOG_FILE_NAME):
            try:
                os.remove(LOG_FILE_NAME)
            except OSError as e:
                print(f"Warning: Could not remove existing log file {LOG_FILE_NAME}: {e}")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE_NAME, mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_strategy_from_config(self):
        """Load the ZigZag Harmonic strategy with parameters from config"""
        try:
            # Get strategy parameters from TRADING_STRATEGY section
            if not self.config.has_section('TRADING_STRATEGY'):
                raise ValueError("TRADING_STRATEGY section not found in config")
            
            strategy_params = dict(self.config['TRADING_STRATEGY'])
            
            # Get simulation settings for additional context
            simulator_settings = {}
            if self.config.has_section('SIMULATOR_SETTINGS'):
                simulator_settings = dict(self.config['SIMULATOR_SETTINGS'])
            
            # Create strategy instance
            self.strategy_instance = TradingStrategy(
                kite_apis_instance=self.k_apis,
                simulation_actual_start_date=date.today(),  # Live trading starts today
                **strategy_params
            )
            
            self.logger.info(f"ZigZag Harmonic Strategy loaded with parameters: {strategy_params}")
            
        except Exception as e:
            self.logger.error(f"Failed to load strategy from config: {e}", exc_info=True)
            raise

    def _load_trader_settings(self):
        """Load live trader specific settings from config"""
        try:
            # Live trader settings (create section if doesn't exist)
            live_trader_section = 'LIVE_TRADER_SETTINGS'
            if not self.config.has_section(live_trader_section):
                self.logger.warning(f"{live_trader_section} section not found, using defaults")
            
            # Polling and timing settings
            self.polling_interval_seconds = self.config.getint(live_trader_section, 'polling_interval_seconds', fallback=POLLING_INTERVAL_SECONDS)
            self.data_lookback_days = self.config.getint(live_trader_section, 'data_lookback_days', fallback=NIFTY_DATA_LOOKBACK_DAYS)
            
            # Trading hours
            self.trading_start_time_str = self.config.get(live_trader_section, 'trading_start_time', fallback='09:20:00')
            self.trading_end_time_str = self.config.get(live_trader_section, 'trading_end_time', fallback='15:00:00')
            self.health_check_start_str = self.config.get(live_trader_section, 'health_check_start_time', fallback='09:15:00')
            self.health_check_end_str = self.config.get(live_trader_section, 'health_check_end_time', fallback='09:20:00')
            
            # Convert to time objects
            self.trading_start_time = dt_time.fromisoformat(self.trading_start_time_str)
            self.trading_end_time = dt_time.fromisoformat(self.trading_end_time_str)
            self.health_check_start_time = dt_time.fromisoformat(self.health_check_start_str)
            self.health_check_end_time = dt_time.fromisoformat(self.health_check_end_str)

            # Token configuration - separate tokens for signal generation and trading
            self.signal_token = self.config.getint(live_trader_section, 'signal_token', fallback=256265)
            self.trading_token = self.config.getint(live_trader_section, 'trading_token', fallback=14536962)
            
            # Trading parameters
            self.trade_quantity = self.config.getint(live_trader_section, 'trade_quantity', fallback=75)
            self.product_type = self.config.get(live_trader_section, 'product_type', fallback='NRML')
            
            self.logger.info(f"Live trader settings loaded. Signal Token: {self.signal_token}, Trading Token: {self.trading_token}")
            
        except Exception as e:
            self.logger.error(f"Failed to load trader settings: {e}", exc_info=True)
            raise

    def _setup_trade_logging(self):
        """Setup CSV trade logging"""
        try:
            # Create trade log file with headers if it doesn't exist
            if not os.path.exists(TRADE_LOG_FILE_NAME):
                with open(TRADE_LOG_FILE_NAME, 'w', newline='') as file:
                    writer = csv.writer(file)
                    headers = [
                        'timestamp', 'trade_id', 'signal_type', 'pattern_name', 
                        'entry_price', 'exit_price', 'quantity', 'pnl', 
                        'entry_order_id', 'exit_order_id', 'entry_reason', 
                        'exit_reason', 'zigzag_d_price', 'tp_price', 'sl_price',
                        'pattern_confidence', 'holding_period_minutes'
                    ]
                    writer.writerow(headers)
            
            self.logger.info(f"Trade logging setup complete: {TRADE_LOG_FILE_NAME}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup trade logging: {e}", exc_info=True)
            raise

    def _system_health_check(self):
        """Perform system health check by testing API connectivity"""
        self.logger.info("Performing system health check...")
        try:
            # Test API connectivity with a simple data fetch
            today_str = date.today().strftime('%Y-%m-%d')
            health_check_data = self.order_manager.kite.historical_data(
                instrument_token=self.signal_token,
                from_date=today_str,
                to_date=today_str,
                interval='minute'
            )
            
            if health_check_data and isinstance(health_check_data, list):
                self.system_healthy = True
                self.logger.info("System health check PASSED. Kite API is responsive.")
            else:
                self.system_healthy = False
                self.logger.warning("System health check FAILED. No data or unexpected response.")
                
        except Exception as e:
            self.system_healthy = False
            self.logger.error(f"System health check FAILED. Exception: {e}", exc_info=True)

    def _manage_trading_session_state(self):
        """Manage trading session state based on time and system health"""
        current_time = datetime.now().time()
        
        # Handle different time windows
        if self.health_check_start_time <= current_time < self.health_check_end_time:
            self._system_health_check()
            self.trade_execution_allowed = False
            self.logger.info("Health check window - trade execution disabled")
            
        elif self.trading_start_time <= current_time < self.trading_end_time:
            if not self.system_healthy:
                self.logger.info("Trading window started but system not healthy. Re-checking...")
                self._system_health_check()
            
            self.trade_execution_allowed = self.system_healthy
            if self.trade_execution_allowed:
                self.logger.debug("Trading window active and system healthy")
            else:
                self.logger.warning("Trading window active but system not healthy")
                
        else:
            self.trade_execution_allowed = False
            if current_time >= self.trading_end_time:
                self.logger.info("Trading session ended for the day")
                self.trading_session_enabled = False

    def _fetch_market_data(self):
        """Fetch market data for the last N days using Kite API directly"""
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=self.data_lookback_days)
            
            end_date_str = end_date.strftime('%Y-%m-%d')
            start_date_str = start_date.strftime('%Y-%m-%d')
            
            self.logger.debug(f"Fetching data from {start_date_str} to {end_date_str} for token {self.signal_token}")
            
            # Use kiteAPIs.getHistoricalData method directly
            historical_data = self.k_apis.getHistoricalData(
                from_date=start_date_str,
                to_date=end_date_str,
                tokens=[self.signal_token],  # Pass as list
                interval='minute'
            )
            
            if historical_data is not None and not historical_data.empty:
                # Convert to the format expected by strategy
                # Ensure 'date' column exists and is datetime
                if 'date' in historical_data.columns:
                    historical_data['date'] = pd.to_datetime(historical_data['date'])
                    historical_data.set_index('date', inplace=True)
                elif 'timestamp' in historical_data.columns:
                    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
                    historical_data.set_index('timestamp', inplace=True)
                    historical_data.index.name = 'date'
                
                # --- NEW: Ignore the latest, incomplete candle ---
                # The strategy should only run on closed, confirmed candles.
                # We slice the dataframe to exclude the last row.
                if len(historical_data) > 1:
                    historical_data = historical_data.iloc[:-1]
                    self.logger.info(f"Ignoring latest candle. Using data up to {historical_data.index[-1]}")

                # Create data_dict format expected by strategy
                data_dict = {
                    'main_interval_data': historical_data.copy(),
                    'one_minute_data': historical_data.copy()
                }
                
                self.logger.debug(f"Prepared {len(historical_data)} rows of historical data for strategy")
                return data_dict
            else:
                self.logger.warning("No historical data fetched from Kite API")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}", exc_info=True)
            return None

    def _generate_signals(self, data_dict):
        """Generate trading signals using the ZigZag Harmonic strategy"""
        try:
            if not data_dict or data_dict['main_interval_data'].empty:
                self.logger.warning("No data available for signal generation")
                return None
            
            # Generate signals using the strategy
            signals_df = self.strategy_instance.generate_signals(data_input_dict=data_dict)
            
            if signals_df.empty:
                self.logger.info("Strategy returned empty signals DataFrame, which is normal if no new patterns or exits occurred.")
                return None
            
            # Filter for all new signals generated in this run
            new_signals = signals_df[signals_df['signal'] != 0]

            if new_signals.empty:
                self.logger.debug("No new entry/exit signals generated in this cycle.")
                return None

            # Get the latest signal from the filtered list
            latest_signal_row = new_signals.iloc[-1]
            latest_timestamp = new_signals.index[-1]
            
            self.logger.debug(f"Latest signal: {latest_signal_row['signal']} at {latest_timestamp}")
            
            return {
                'signals_df': signals_df, # Pass the full DF for context if needed
                'latest_signal': latest_signal_row['signal'],
                'latest_timestamp': latest_timestamp,
                'latest_row': latest_signal_row
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}", exc_info=True)
            return None

    def _execute_trades(self, signal_data):
        """
        Execute trades based on the latest signal, but only if it's recent.
        This function checks only the most recent signal and acts if it's within a 5-minute window.
        """
        try:
            signals_df = signal_data.get('signals_df')
            if signals_df is None or signals_df.empty:
                return

            # Filter for all rows with a non-zero signal
            signals_with_action = signals_df[signals_df['signal'] != 0]

            if signals_with_action.empty:
                self.logger.debug("No signals with action found in the latest data.")
                return

            # Get the single most recent signal
            last_signal_row = signals_with_action.iloc[-1]
            last_signal_timestamp = signals_with_action.index[-1]

            # Avoid re-processing the same signal on subsequent ticks
            if self.last_signal_timestamp and self.last_signal_timestamp >= last_signal_timestamp:
                self.logger.debug(f"Signal at {last_signal_timestamp} has already been processed. Skipping.")
                return

            # Check if the signal is too old to act upon
            current_time = pd.Timestamp.now(tz='Asia/Kolkata') # Ensure current time is tz-aware
            
            # The timestamp from the data is likely already tz-aware. If so, convert; if not, localize.
            if last_signal_timestamp.tzinfo is None:
                signal_ts_for_comparison = last_signal_timestamp.tz_localize('Asia/Kolkata')
            else:
                signal_ts_for_comparison = last_signal_timestamp.tz_convert('Asia/Kolkata')

            time_difference = current_time - signal_ts_for_comparison

            if time_difference > timedelta(minutes=5):
                self.logger.warning(f"IGNORING STALE SIGNAL from {last_signal_timestamp} "
                                  f"({time_difference.total_seconds() / 60:.2f} mins ago). Current time: {current_time}")
                self.last_signal_timestamp = last_signal_timestamp  # Mark as considered
                return
            
            # If we reach here, the signal is fresh and new. Process it.
            self.logger.info(f"Processing fresh signal from {last_signal_timestamp} ({time_difference.total_seconds():.2f}s old).")
            
            signal = last_signal_row['signal']
            pattern_tag = last_signal_row.get('pattern_tag', '')
            current_price = last_signal_row['close']
            
            # Send Telegram notification for the fresh signal
            if signal == 1:
                if 'bull' in pattern_tag.lower() or 'long' in pattern_tag.lower():
                     signal_type_text = "BUY/LONG SIGNAL"
                elif 'bear' in pattern_tag.lower() or 'short' in pattern_tag.lower():
                     signal_type_text = "SELL/SHORT SIGNAL"
                else:
                     signal_type_text = "ENTRY SIGNAL (Check Direction)"
            else: # signal == -1
                signal_type_text = "EXIT SIGNAL"

            self.order_manager.send_telegram_message(
                f"🔔 FRESH STRATEGY SIGNAL 🔔\n"
                f"Type: {signal_type_text}\n"
                f"Price: {current_price:.2f}\n"
                f"Pattern: {pattern_tag[:70]}...\n"
                f"Time: {last_signal_timestamp.strftime('%H:%M:%S')}"
            )
            self.logger.info(f"Telegram notification sent for fresh signal: {signal_type_text} at {current_price:.2f}")

            # Process EXIT signals
            if signal == -1:
                self.logger.info(f"Processing EXIT signal. Is any trade active? {self.is_any_trade_active}")
                if self.is_any_trade_active:
                    trades_to_close_ids = list(self.active_trades.keys())
                    closed_count = 0
                    for trade_id in trades_to_close_ids:
                        trade_details = self.active_trades.get(trade_id)
                        # Exit if trade is ACTIVE or PENDING (to handle unconfirmed entries)
                        if trade_details and trade_details.get('status') in ['ACTIVE', 'PENDING_CONFIRMATION']:
                            self.logger.info(f"Exit signal received for trade {trade_id} with status {trade_details.get('status')}. Closing it.")
                            self._close_trade(trade_details, current_price, f"STRATEGY_EXIT: {pattern_tag}")
                            del self.active_trades[trade_id]
                            closed_count += 1
                    
                    self.is_any_trade_active = len(self.active_trades) > 0
                    if closed_count > 0:
                        self.logger.info(f"Strategy exit signal processed. Closed {closed_count} trades at {current_price:.2f}")
                else:
                    self.logger.info("Exit signal received, but no trade is active. Ignoring.")

            # Process ENTRY signals
            elif signal == 1:
                self.logger.info(f"Processing ENTRY signal. Is any trade active? {self.is_any_trade_active}")
                if not self.is_any_trade_active:
                    if 'entry' in pattern_tag.lower():
                        self._initiate_new_trade(last_signal_row, last_signal_timestamp, pattern_tag)
                    else:
                        self.logger.warning(f"Entry signal (1) found, but pattern tag '{pattern_tag}' does not contain 'entry'. Ignoring entry.")
                else:
                    self.logger.info("Entry signal received, but a trade is already active. Ignoring.")
            
            # Mark this signal as processed
            self.last_signal_timestamp = last_signal_timestamp

        except Exception as e:
            self.logger.error(f"Error in _execute_trades: {e}", exc_info=True)

    def _initiate_new_trade(self, signal_row, timestamp, pattern_tag):
        """Initiate a new trade based on signal"""
        try:
            current_price = signal_row['close']
            
            # Extract pattern information
            pattern_name = self._extract_pattern_name(pattern_tag)
            
            # --- FIX: Determine trade direction from the reliable pattern_mode ---
            pattern_mode = signal_row.get('pattern_mode', 0) # 1 for bull/long, -1 for bear/short
            if pattern_mode == 1:
                trade_direction = 'LONG'
            elif pattern_mode == -1:
                trade_direction = 'SHORT'
            else:
                self.logger.error(f"Could not determine trade direction from pattern_mode: {pattern_mode}. Aborting trade initiation.")
                return

            # Generate unique trade ID
            trade_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trade_direction}_{pattern_name.replace(' ', '')}"
            
            self.logger.info(f"Initiating {trade_direction} trade: {pattern_name} at {current_price:.2f}")
            
            # Determine transaction type
            transaction_type = (self.order_manager.kite.TRANSACTION_TYPE_BUY 
                              if trade_direction == 'LONG' 
                              else self.order_manager.kite.TRANSACTION_TYPE_SELL)
            
            # Attempt order placement with retries
            order_id = None
            for attempt in range(1, MAX_ENTRY_ORDER_RETRIES + 1):
                try:
                    order_id = self.order_manager.place_market_order_live(
                        tradingsymbol=f"NIFTY{datetime.now().strftime('%y%b').upper()}FUT",  # Current month future
                        exchange=self.order_manager.kite.EXCHANGE_NFO,
                        transaction_type=transaction_type,
                        quantity=self.trade_quantity,
                        product=self.product_type
                    )
                    
                    if order_id:
                        self.logger.info(f"Order placed successfully on attempt {attempt}. Order ID: {order_id}")
                        break
                    else:
                        self.logger.warning(f"Order placement returned None on attempt {attempt}")
                        
                except Exception as order_error:
                    self.logger.warning(f"Order placement attempt {attempt} failed: {order_error}")
                    
                # If not the last attempt, wait before retrying
                if attempt < MAX_ENTRY_ORDER_RETRIES:
                    self.logger.info(f"Waiting {ENTRY_ORDER_RETRY_DELAY_SECONDS}s before retry...")
                    time.sleep(ENTRY_ORDER_RETRY_DELAY_SECONDS)
            
            if not order_id:
                self.logger.error(f"Failed to place order for {trade_direction} trade after {MAX_ENTRY_ORDER_RETRIES} attempts")
                return
            
            # Send telegram notification
            self.order_manager.send_telegram_message(
                f"LiveTrader: {trade_direction} ENTRY order placed\n"
                f"Pattern: {pattern_name}\n"
                f"Price: {current_price:.2f}\n"
                f"Order ID: {order_id}"
            )
            
            # Calculate TP and SL levels from signal row
            tp_price = signal_row.get('pattern_tp1_price', np.nan)
            sl_price = signal_row.get('pattern_sl1_price', np.nan)
            
            # Store trade details
            trade_details = {
                'trade_id': trade_id,
                'direction': trade_direction,
                'pattern_name': pattern_name,
                'entry_timestamp': timestamp,
                'entry_price': current_price,
                'quantity': self.trade_quantity,
                'order_id': order_id,
                'status': 'PENDING_CONFIRMATION',
                'tp_price': tp_price,
                'sl_price': sl_price,
                'pattern_tag': pattern_tag
            }
            
            # Log the action
            log_message = f"{timestamp} {trade_direction:<4} | ENTRY: {pattern_tag} at {current_price:.2f}"
            self.action_logger.info(log_message)

            self.active_trades[trade_id] = trade_details
            self.is_any_trade_active = True
            self.trade_count_today += 1
            
            self.logger.info(f"Trade initiated: {trade_id}")
            
        except Exception as e:
            self.logger.error(f"Error initiating new trade: {e}", exc_info=True)

    def _monitor_positions(self):
        """Monitor existing positions and handle exits"""
        if not self.active_trades:
            return
        
        try:
            current_time = datetime.now()
            trades_to_close = []
            
            # Fetch the latest candle data once for this monitoring cycle
            latest_candle = self._get_latest_candle_data()
            if latest_candle is None:
                self.logger.warning("Could not get latest candle data for monitoring positions.")
                return

            for trade_id, trade_details in self.active_trades.items():
                try:
                    # Check trade status and update if needed
                    if trade_details['status'] == 'PENDING_CONFIRMATION':
                        self._check_order_confirmation(trade_details)
                    
                    # Skip monitoring if trade not yet confirmed
                    if trade_details['status'] != 'ACTIVE':
                        continue
                    
                    # Check exit conditions using the full candle data
                    exit_reason = self._check_exit_conditions(trade_details, latest_candle, current_time)
                    
                    if exit_reason:
                        # Use the close of the candle for the exit price record
                        self._close_trade(trade_details, latest_candle['close'], exit_reason)
                        trades_to_close.append(trade_id)
                        
                except Exception as e:
                    self.logger.error(f"Error monitoring trade {trade_id}: {e}", exc_info=True)
            
            # Remove closed trades
            for trade_id in trades_to_close:
                if trade_id in self.active_trades:
                    del self.active_trades[trade_id]
            
            # Update active trade status
            self.is_any_trade_active = len(self.active_trades) > 0
            
        except Exception as e:
            self.logger.error(f"Error in _monitor_positions: {e}", exc_info=True)

    def _check_order_confirmation(self, trade_details):
        """Check if order has been confirmed"""
        try:
            order_id = trade_details['order_id']
            order_history = self.order_manager.get_order_history_live(order_id)
            
            if order_history:
                latest_status = order_history[-1]
                if latest_status['status'] == self.order_manager.kite.STATUS_COMPLETE:
                    avg_price = latest_status.get('average_price', trade_details['entry_price'])
                    trade_details['entry_price'] = avg_price
                    trade_details['status'] = 'ACTIVE'
                    self.logger.info(f"Trade {trade_details['trade_id']} confirmed at {avg_price:.2f}")
                    
                    # Send telegram confirmation
                    self.order_manager.send_telegram_message(
                        f"LiveTrader: Trade CONFIRMED\n"
                        f"ID: {trade_details['trade_id']}\n"
                        f"Entry: {avg_price:.2f}\n"
                        f"TP: {trade_details['tp_price']:.2f}\n"
                        f"SL: {trade_details['sl_price']:.2f}"
                    )
                    
                elif latest_status['status'] in [self.order_manager.kite.STATUS_REJECTED, self.order_manager.kite.STATUS_CANCELLED]:
                    trade_details['status'] = 'FAILED'
                    self.logger.error(f"Trade {trade_details['trade_id']} failed: {latest_status.get('status_message')}")
                    
        except Exception as e:
            self.logger.error(f"Error checking order confirmation: {e}", exc_info=True)

    def _get_latest_candle_data(self):
        """Get the latest full candle data (OHLC) for the signal instrument."""
        try:
            today_str = date.today().strftime('%Y-%m-%d')
            price_data = self.order_manager.kite.historical_data(
                instrument_token=self.signal_token,
                from_date=today_str,
                to_date=today_str,
                interval='minute'
            )
            
            if price_data and len(price_data) > 0:
                # Returns the last dictionary in the list, which contains ohlc
                return price_data[-1]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest candle data: {e}", exc_info=True)
            return None

    def _check_exit_conditions(self, trade_details, latest_candle, current_time):
        """Check if any exit conditions are met using full candle data."""
        try:
            direction = trade_details['direction']
            tp_price = trade_details['tp_price']
            sl_price = trade_details['sl_price']
            
            high_price = latest_candle['high']
            low_price = latest_candle['low']
            
            # Check TP/SL conditions based on direction using high/low
            if direction == 'LONG':
                if not pd.isna(tp_price) and high_price >= tp_price:
                    return f"TP_MONITOR(H>={tp_price:.2f})"
                elif not pd.isna(sl_price) and low_price <= sl_price:
                    return f"SL_MONITOR(L<={sl_price:.2f})"
            else:  # SHORT
                if not pd.isna(tp_price) and low_price <= tp_price:
                    return f"TP_MONITOR(L<={tp_price:.2f})"
                elif not pd.isna(sl_price) and high_price >= sl_price:
                    return f"SL_MONITOR(H>={sl_price:.2f})"
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}", exc_info=True)
            return None

    def _close_trade(self, trade_details, exit_price, exit_reason):
        """Close an active trade"""
        try:
            trade_id = trade_details['trade_id']
            direction = trade_details['direction']
            
            self.logger.info(f"Closing trade {trade_id} with exit price {exit_price} due to {exit_reason}")
            self.action_logger.info(f"{datetime.now()} {direction:<4} | EXIT: {trade_details['pattern_tag']} at {exit_price:.2f} due to {exit_reason}")
            
            # Place exit order using trading_token
            exit_transaction_type = (self.order_manager.kite.TRANSACTION_TYPE_SELL 
                                   if direction == 'LONG' 
                                   else self.order_manager.kite.TRANSACTION_TYPE_BUY)
            
            exit_order_id = self.order_manager.place_market_order_live(
                tradingsymbol=f"NIFTY{datetime.now().strftime('%y%b').upper()}FUT",
                exchange=self.order_manager.kite.EXCHANGE_NFO,
                transaction_type=exit_transaction_type,
                quantity=self.trade_quantity,
                product=self.product_type
            )
            
            # Calculate PnL
            pnl = 0
            entry_price = trade_details['entry_price']
            if direction == 'LONG':
                pnl_per_unit = exit_price - entry_price
            else:
                pnl_per_unit = entry_price - exit_price
            
            total_pnl = pnl_per_unit * self.trade_quantity
            self.daily_pnl += total_pnl
            
            # Calculate holding period
            entry_time = trade_details['entry_timestamp']
            holding_period = (datetime.now() - entry_time).total_seconds() / 60  # minutes
            
            # Prepare trade log data
            trade_log_data = {
                'timestamp': datetime.now(),
                'trade_id': trade_id,
                'signal_type': direction,
                'pattern_name': trade_details['pattern_name'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': self.trade_quantity,
                'pnl': total_pnl,
                'entry_order_id': trade_details.get('order_id', 'N/A'),
                'exit_order_id': exit_order_id or 'N/A',
                'entry_reason': trade_details['pattern_tag'],
                'exit_reason': exit_reason,
                'zigzag_d_price': 'N/A',  # Could extract from pattern data
                'tp_price': trade_details.get('tp_price', 'N/A'),
                'sl_price': trade_details.get('sl_price', 'N/A'),
                'pattern_confidence': 'N/A',
                'holding_period_minutes': holding_period
            }
            
            # Log the trade
            self._log_trade(trade_log_data)
            
            # Send telegram notification
            self.order_manager.send_telegram_message(
                f"LiveTrader: Trade CLOSED\n"
                f"ID: {trade_id}\n"
                f"Reason: {exit_reason}\n"
                f"PnL: {total_pnl:.2f}\n"
                f"Exit Price: {exit_price:.2f}"
            )
            
            self.logger.info(f"Trade {trade_id} closed. PnL: {total_pnl:.2f}, Daily PnL: {self.daily_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error closing trade: {e}", exc_info=True)

    def _log_trade(self, trade_data):
        """Log trade details to CSV file"""
        try:
            with open(TRADE_LOG_FILE_NAME, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'timestamp', 'trade_id', 'signal_type', 'pattern_name', 
                    'entry_price', 'exit_price', 'quantity', 'pnl', 
                    'entry_order_id', 'exit_order_id', 'entry_reason', 
                    'exit_reason', 'zigzag_d_price', 'tp_price', 'sl_price',
                    'pattern_confidence', 'holding_period_minutes'
                ])
                writer.writerow(trade_data)
            
            self.logger.info(f"Trade logged to {TRADE_LOG_FILE_NAME}")
            
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}", exc_info=True)

    def _extract_pattern_name(self, pattern_tag):
        """Extract pattern name from pattern tag"""
        try:
            # Simple extraction - look for common pattern names
            pattern_names = ['Bat', 'Butterfly', 'Gartley', 'Crab', 'Shark', 'ABCD', '5-O', 'Wolf', 'H&S']
            for name in pattern_names:
                if name.lower() in pattern_tag.lower():
                    return name
            return 'Unknown Pattern'
        except:
            return 'Unknown Pattern'

    def _extract_trade_direction(self, pattern_tag):
        """Extract trade direction from pattern tag"""
        try:
            if 'long' in pattern_tag.lower():
                return 'LONG'
            elif 'short' in pattern_tag.lower():
                return 'SHORT'
            elif 'bull' in pattern_tag.lower():
                return 'LONG'
            elif 'bear' in pattern_tag.lower():
                return 'SHORT'
            return 'LONG'  # Default
        except:
            return 'LONG'

    def run_live_session(self):
        """Main trading loop"""
        self.logger.info("--- Starting Live Trading Session (ZigZag Harmonic) ---")
        
        while self.trading_session_enabled:
            try:
                loop_start_time = time.time()
                
                # 1. Manage trading session state
                self._manage_trading_session_state()
                
                if not self.trading_session_enabled:
                    break
                
                # 2. Fetch market data
                data_dict = self._fetch_market_data()
                
                if data_dict:
                    # 3. Generate signals
                    signal_data = self._generate_signals(data_dict)
                    
                    if signal_data:
                        # 4. Monitor existing positions
                        self._monitor_positions()
                        
                        # 5. Execute new trades if conditions are met
                        if self.trade_execution_allowed:
                            self._execute_trades(signal_data)
                        
                        # Log current status
                        latest_close = data_dict['main_interval_data']['close'].iloc[-1]
                        self.logger.info(f"NIFTY: {latest_close:.2f}, Signal: {signal_data['latest_signal']}, "
                                       f"Active Trades: {len(self.active_trades)}, Daily PnL: {self.daily_pnl:.2f}")
                
                # Calculate sleep time to maintain polling interval
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, self.polling_interval_seconds - loop_duration)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self.logger.warning(f"Loop took {loop_duration:.2f}s, longer than polling interval {self.polling_interval_seconds}s")
                
            except KeyboardInterrupt:
                self.logger.warning("Keyboard interrupt received. Shutting down...")
                self.trading_session_enabled = False
                
            except Exception as e:
                self.logger.error(f"Unexpected error in main trading loop: {e}", exc_info=True)
                time.sleep(5)  # Brief pause before continuing
        
        self.logger.info("--- Live Trading Session Ended ---")

# --- Main Execution ---
if __name__ == '__main__':
    # Determine config file path
    current_dir = os.path.dirname(__file__)
    config_file = os.path.join(current_dir, 'trading_config.ini')
    
    if not os.path.exists(config_file):
        config_file_alt = os.path.join(os.path.dirname(current_dir), 'trading_config.ini')
        if os.path.exists(config_file_alt):
            config_file = config_file_alt
        else:
            print(f"FATAL: trading_config.ini not found at {config_file} or parent directory. Exiting.")
            sys.exit(1)
    
    print(f"Using configuration file: {config_file}")
    
    try:
        trader = LiveTrader(config_file_path=config_file)
        trader.run_live_session()
    except FileNotFoundError as e:
        print(f"FATAL: Could not start LiveTrader due to missing file: {e}")
    except ValueError as e:
        print(f"FATAL: Could not start LiveTrader due to configuration error: {e}")
    except SystemExit as e:
        print(f"STOPPING: {e}")
    except Exception as e:
        print(f"FATAL: An unexpected error prevented LiveTrader from starting: {e}")
        import traceback
        traceback.print_exc() 