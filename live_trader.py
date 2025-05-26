import logging
import time
from datetime import datetime, date, timedelta, time as dt_time
import configparser
import os
import pandas as pd
import numpy as np # For np.nan if needed
import sys # For sys.exit()

# Assuming myKiteLib.py and trading_strategies.py are accessible
from myKiteLib import OrderPlacement # For live trading operations
from trading_strategies import DataPrep, DonchianBreakoutStrategy # Add other strategies as needed

# --- Constants --- #
# These could also be moved to a separate constants file or into config
LOG_FILE_NAME = "live_trader.log"
NIFTY_DATA_FETCH_CANDLES = 60 # Number of 1-min Nifty candles to fetch for strategy calculation (e.g., 20 for Donchian + buffer)
MAX_ENTRY_ORDER_RETRIES = 2 # Max number of retries after initial attempt (total 3 attempts)
ENTRY_ORDER_RETRY_DELAY_SECONDS = 3

class LiveTrader:
    def __init__(self, config_file_path: str):
        self.config_file_path = config_file_path
        self.config = configparser.ConfigParser()
        if not os.path.exists(self.config_file_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file_path}")
        self.config.read(self.config_file_path)
        self._setup_logging()
        self.logger.info("--- LiveTrader Initializing ---")

        # Initialize components
        try:
            self.order_manager = OrderPlacement()
            # It's crucial to ensure the Kite API session is active.
            # The OrderPlacement class inherits system_initialization, which has init_trading()
            self.logger.info("OrderPlacement initialized. Attempting to initialize trading session...")
            self.order_manager.init_trading() # This handles token validation and instrument download
            self.logger.info("Kite trading session initialized successfully via OrderPlacement.")

            self.data_prep = DataPrep()
            self.logger.info("DataPrep initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize core components (OrderManager/DataPrep): {e}", exc_info=True)
            raise SystemExit(f"Core component initialization failed: {e}")

        # Load strategy from config
        self._load_strategy_from_config()

        # State variables
        self.active_trade_details = {} # Stores details of the currently active trade
        self.is_trade_active = False
        self.trading_session_enabled = True # Overall kill switch for the trading session
        self.system_healthy = False # Tracks Kite API health
        self.trade_execution_allowed = False # Flag set by health check and time checks

        # Load trader settings from config
        self.nifty_index_token = self.config.getint('SIMULATOR_SETTINGS', 'index_token') # Reusing simulator setting for Nifty token
        self.polling_interval_seconds = self.config.getint('LIVE_TRADER_SETTINGS', 'polling_interval_seconds', fallback=60)
        self.trading_start_time_str = self.config.get('LIVE_TRADER_SETTINGS', 'trading_start_time', fallback='09:20:00')
        self.trading_end_time_str = self.config.get('LIVE_TRADER_SETTINGS', 'trading_end_time', fallback='15:00:00')
        self.health_check_start_str = self.config.get('LIVE_TRADER_SETTINGS', 'health_check_start_time', fallback='09:15:00')
        self.health_check_end_str = self.config.get('LIVE_TRADER_SETTINGS', 'health_check_end_time', fallback='09:20:00')
        
        self.trading_start_time = dt_time.fromisoformat(self.trading_start_time_str)
        self.trading_end_time = dt_time.fromisoformat(self.trading_end_time_str)
        self.health_check_start_time = dt_time.fromisoformat(self.health_check_start_str)
        self.health_check_end_time = dt_time.fromisoformat(self.health_check_end_str)

        # Option and trade parameters (will be from the selected strategy's config section)
        self.option_type = self.config.get(self.selected_strategy_config_section, 'option_type', fallback='CE').upper()
        self.trade_units = self.config.getint(self.selected_strategy_config_section, 'trade_units')
        self.profit_target_pct = self.config.getfloat(self.selected_strategy_config_section, 'profit_target_pct')
        self.stop_loss_pct = self.config.getfloat(self.selected_strategy_config_section, 'stop_loss_pct')
        self.max_holding_period_minutes = self.config.getint(self.selected_strategy_config_section, 'max_holding_period_minutes', fallback=30) # Default to 30 mins if not specified

        self.option_lot_size = None # Will be fetched when an option is chosen
        self.trade_quantity_actual = None # Will be trade_units * option_lot_size

        self.logger.info(f"LiveTrader Initialized. Polling: {self.polling_interval_seconds}s. Trading Window: {self.trading_start_time_str}-{self.trading_end_time_str}")
        self.logger.info(f"Strategy: {type(self.strategy_obj).__name__} with params from {self.selected_strategy_config_section}")
        self.logger.info(f"Option Type: {self.option_type}, Base Units (Lots): {self.trade_units}, SL: {self.stop_loss_pct*100}%, TP: {self.profit_target_pct*100}%, Max Hold: {self.max_holding_period_minutes} min")

    def _setup_logging(self):
        # Overwrite log file on each run
        if os.path.exists(LOG_FILE_NAME):
            try:
                os.remove(LOG_FILE_NAME)
            except OSError as e:
                print(f"Error removing existing log file {LOG_FILE_NAME}: {e}")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE_NAME, mode='w'), # 'w' to overwrite
                logging.StreamHandler() # Also log to console
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_strategy_from_config(self):
        self.selected_strategy_config_section = self.config.get(
            'LIVE_TRADER_SETTINGS',
            'active_strategy_config_section',
            fallback='STRATEGY_CONFIG_DonchianStandard' # Default if not specified
        )
        self.logger.info(f"Loading strategy from config section: {self.selected_strategy_config_section}")

        if not self.config.has_section(self.selected_strategy_config_section):
            self.logger.error(f"Strategy configuration section '{self.selected_strategy_config_section}' not found in config.")
            raise ValueError(f"Strategy config section '{self.selected_strategy_config_section}' not found.")

        strategy_class_name = self.config.get(self.selected_strategy_config_section, 'strategy_class_name')
        strategy_indicator_params = {}

        if strategy_class_name == 'DonchianBreakoutStrategy':
            strategy_indicator_params['length'] = self.config.getint(self.selected_strategy_config_section, 'length')
            global NIFTY_DATA_FETCH_CANDLES # Allow modification of global
            NIFTY_DATA_FETCH_CANDLES = max(NIFTY_DATA_FETCH_CANDLES, strategy_indicator_params['length'] + 10) # Ensure enough data for lookback
            strategy_indicator_params['exit_option'] = self.config.getint(self.selected_strategy_config_section, 'exit_option')
            self.strategy_obj = DonchianBreakoutStrategy(**strategy_indicator_params)
        # Add elif for other strategy types here
        # elif strategy_class_name == 'AnotherStrategy':
        #     self.strategy_obj = AnotherStrategy(...)
        else:
            self.logger.error(f"Unsupported strategy_class_name '{strategy_class_name}' in config.")
            raise ValueError(f"Unsupported strategy class: {strategy_class_name}")
        
        self.logger.info(f"Strategy '{type(self.strategy_obj).__name__}' loaded with params: {strategy_indicator_params}")

    # --- System Health and Time Management ---
    def _system_health_check(self):
        self.logger.info("Performing system health check...")
        try:
            # Attempt to fetch historical data for NIFTY as a health check
            # Use a very short period to minimize data transfer
            today_str = date.today().strftime('%Y-%m-%d')
            health_check_data = self.order_manager.kite.historical_data(
                instrument_token=self.nifty_index_token,
                from_date=today_str, # From today
                to_date=today_str,   # To today
                interval='minute'
            )
            if health_check_data is not None and isinstance(health_check_data, list): # historical_data returns a list of dicts
                self.system_healthy = True
                self.logger.info("System health check PASSED. Kite API is responsive.")
            else:
                self.system_healthy = False
                self.logger.warning("System health check FAILED. No data or unexpected response from Kite API.")
        except Exception as e:
            self.system_healthy = False
            self.logger.error(f"System health check FAILED. Exception: {e}", exc_info=True)

    def _manage_trading_session_state(self):
        current_time_now = datetime.now().time()
        self.logger.debug(f"Current time: {current_time_now}, System Healthy: {self.system_healthy}, Trade Active: {self.is_trade_active}")

        if self.health_check_start_time <= current_time_now < self.health_check_end_time:
            self._system_health_check()
            self.trade_execution_allowed = False # No trades during health check window
            self.logger.info(f"Inside health check window ({self.health_check_start_time_str} - {self.health_check_end_time_str}). Trade execution disallowed.")
        elif self.trading_start_time <= current_time_now < self.trading_end_time:
            if not self.system_healthy:
                # If system wasn't healthy before trading window, re-check once when window starts
                self.logger.info("Trading window started, but system was not healthy. Re-checking health...")
                self._system_health_check() 
            
            self.trade_execution_allowed = self.system_healthy
            if self.trade_execution_allowed:
                self.logger.info(f"Inside trading window ({self.trading_start_time_str} - {self.trading_end_time_str}) and system healthy. Trade execution allowed.")
            else:
                self.logger.warning(f"Inside trading window but system not healthy. Trade execution disallowed.")
        else:
            self.trade_execution_allowed = False
            self.logger.info(f"Outside trading hours/health check window. Trade execution disallowed.")
            if current_time_now >= self.trading_end_time:
                self.logger.info("Trading session ended for the day based on time.")
                self.trading_session_enabled = False # Stop the main loop

    # --- Main Loop --- #
    def run_live_session(self):
        self.logger.info("--- Starting Live Trading Session ---")
        while self.trading_session_enabled:
            try:
                self._manage_trading_session_state()

                if not self.trading_session_enabled: # Check if _manage_trading_session_state decided to end session
                    break

                # 1. Fetch NIFTY data
                today_str = date.today().strftime('%Y-%m-%d')
                nifty_data = self.order_manager.kite.historical_data(
                    instrument_token=self.nifty_index_token,
                    from_date=today_str,
                    to_date=today_str,
                    interval='minute'
                )
                
                if nifty_data:
                    df = pd.DataFrame(nifty_data)
                    # Debug log to check data structure
                    self.logger.info(f"Raw data columns: {df.columns.tolist()}")
                    self.logger.info(f"First row date type: {type(df['date'].iloc[0]) if 'date' in df.columns else 'No date column'}")
                    
                    # Convert date column to datetime if it exists
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        self.logger.info(f"After conversion - First row date type: {type(df['date'].iloc[0])}")
                    
                    # 2. Calculate stats & Generate signals
                    df_with_stats = self.data_prep.calculate_statistics(df.copy(), donchian_length=self.strategy_obj.length)
                    if not df_with_stats.empty:
                        latest_row = df_with_stats.iloc[-1]
                        self.logger.info(f"Latest row date type: {type(latest_row['date']) if 'date' in latest_row else 'No date in latest row'}")
                        
                        prev_row = df_with_stats.iloc[-2] if len(df_with_stats) > 1 else None
                        prev_close = prev_row['close'] if prev_row is not None else None
                        prev_close_str = f"{prev_close:.2f}" if prev_close is not None else "N/A"
                        self.logger.info(f"NIFTY Stats - Close: {latest_row['close']:.2f}, Previous Donchian Upper: {latest_row['don_upper_prev']:.2f}, Previous Close: {prev_close_str}")
                        
                        # Generate signals
                        df_with_signals = self.strategy_obj.generate_signals(df_with_stats.copy())
                        latest_signal = df_with_signals.iloc[-1]['signal']
                        
                        # 3. If trade active: monitor
                        if self.is_trade_active:
                            self._monitor_active_trade(latest_signal, latest_row)
                        # 4. If no trade active & BUY signal & trade_execution_allowed: initiate
                        elif latest_signal == 1 and self.trade_execution_allowed:
                            # Use the timestamp from the data
                            signal_time = latest_row['date'] if 'date' in latest_row else datetime.now()
                            self.logger.info(f"Signal time type before trade initiation: {type(signal_time)}")
                            self._initiate_new_trade(signal_time, latest_row['close'])

                self.logger.info(f"Looping... Trade Execution Allowed: {self.trade_execution_allowed}, Trade Active: {self.is_trade_active}")
                time.sleep(self.polling_interval_seconds) 

            except KeyboardInterrupt:
                self.logger.warning("Keyboard interrupt received. Shutting down live trader...")
                self.trading_session_enabled = False
            except Exception as e:
                self.logger.error(f"An unexpected error occurred in the main trading loop: {e}", exc_info=True)
                self.trading_session_enabled = False
                time.sleep(5) 

        self.logger.info("--- Live Trading Session Ended ---")

    def _find_live_option_token(self, nifty_price_at_signal, nifty_signal_time):
        """Find a suitable option token based on current NIFTY price and expiry using SQL queries.
        
        Args:
            nifty_price_at_signal: Current NIFTY price when signal was generated
            nifty_signal_time: Time when the signal was generated
            
        Returns:
            tuple: (tradingsymbol, token, lot_size) or None if no suitable option found
        """
        try:
            # Debug log for input parameters
            self.logger.info(f"Finding option token - Signal time type: {type(nifty_signal_time)}, Value: {nifty_signal_time}")
            
            # Get database connection from kiteAPIs
            conn = self.order_manager.k_apis.startKiteSession.con
            if not conn or not conn.is_connected():
                self.logger.error("Database connection not available")
                return None

            # Ensure nifty_signal_time is a datetime object
            if not isinstance(nifty_signal_time, datetime):
                self.logger.error(f"Invalid signal time type: {type(nifty_signal_time)}. Expected datetime.")
                return None

            # Build query based on option type
            if self.option_type == 'CE':
                query = """
                WITH last_expiry_month AS (
                    SELECT MAX(expiry) AS last_expiry_month
                    FROM kiteConnect.instruments_zerodha
                    WHERE name = 'NIFTY'
                      AND instrument_type = 'CE'
                      AND EXTRACT(MONTH FROM expiry) = EXTRACT(MONTH FROM %(signal_date)s)
                      AND EXTRACT(YEAR FROM expiry) = EXTRACT(YEAR FROM %(signal_date)s)
                ),
                filtered_options AS (
                    SELECT a.instrument_token, a.tradingsymbol, a.lot_size, a.strike,
                           ROW_NUMBER() OVER (ORDER BY a.strike ASC) AS rnum
                    FROM kiteConnect.instruments_zerodha a
                    INNER JOIN last_expiry_month b ON a.expiry = b.last_expiry_month
                    WHERE a.name = 'NIFTY'
                      AND a.instrument_type = 'CE'
                      AND a.strike >= %(strike_price)s
                )
                SELECT instrument_token, tradingsymbol, lot_size, strike
                FROM filtered_options 
                WHERE rnum = 1;
                """
            else:  # PE
                query = """
                WITH last_expiry_month AS (
                    SELECT MAX(expiry) AS last_expiry_month
                    FROM kiteConnect.instruments_zerodha
                    WHERE name = 'NIFTY'
                      AND instrument_type = 'PE'
                      AND EXTRACT(MONTH FROM expiry) = EXTRACT(MONTH FROM %(signal_date)s)
                      AND EXTRACT(YEAR FROM expiry) = EXTRACT(YEAR FROM %(signal_date)s)
                ),
                filtered_options AS (
                    SELECT a.instrument_token, a.tradingsymbol, a.lot_size, a.strike,
                           ROW_NUMBER() OVER (ORDER BY a.strike DESC) AS rnum
                    FROM kiteConnect.instruments_zerodha a
                    INNER JOIN last_expiry_month b ON a.expiry = b.last_expiry_month
                    WHERE a.name = 'NIFTY'
                      AND a.instrument_type = 'PE'
                      AND a.strike <= %(strike_price)s
                )
                SELECT instrument_token, tradingsymbol, lot_size, strike
                FROM filtered_options 
                WHERE rnum = 1;
                """

            params = {
                'signal_date': nifty_signal_time.date(),
                'strike_price': float(nifty_price_at_signal)
            }
            
            self.logger.info(f"SQL params - signal_date: {params['signal_date']}, strike_price: {params['strike_price']}")

            option_df = pd.read_sql_query(query, conn, params=params)
            
            if option_df.empty:
                self.logger.error(f"No {self.option_type} option found for NIFTY at {nifty_price_at_signal:.2f}, time {nifty_signal_time}")
                return None

            # Extract option details
            option_token = int(option_df['instrument_token'].iloc[0])
            option_symbol = option_df['tradingsymbol'].iloc[0]
            option_lot_size = int(option_df['lot_size'].iloc[0])
            strike_price = float(option_df['strike'].iloc[0])

            self.logger.info(f"Selected {self.option_type} option: {option_symbol} (Strike: {strike_price}, "
                           f"Lot Size: {option_lot_size})")
            
            return (option_symbol, option_token, option_lot_size)
            
        except Exception as e:
            self.logger.error(f"Error finding suitable option: {e}", exc_info=True)
            return None

    def _initiate_new_trade(self, nifty_signal_time: datetime, nifty_price_at_signal: float):
        # Safety check for datetime
        if not isinstance(nifty_signal_time, datetime):
            self.logger.error(f"Invalid signal time type in _initiate_new_trade: {type(nifty_signal_time)}. Expected datetime.")
            return
            
        self.logger.info(f"Attempting to initiate new {self.option_type} trade based on NIFTY signal at {nifty_signal_time}, Price: {nifty_price_at_signal:.2f}")
        
        option_selection = self._find_live_option_token(nifty_price_at_signal, nifty_signal_time)
        if not option_selection:
            self.logger.warning("Could not find a suitable live option. Skipping trade initiation.")
            return

        option_symbol, option_token, option_lot_size = option_selection
        self.option_lot_size = option_lot_size 
        self.trade_quantity_actual = self.trade_units * self.option_lot_size

        self.logger.info(f"Attempting to place MARKET BUY order for {self.trade_quantity_actual} units of {option_symbol} ({self.option_type}).")
        
        product_type = self.config.get('LIVE_TRADER_SETTINGS', 'product_type', fallback=self.order_manager.kite.PRODUCT_MIS)
        entry_order_id = None
        
        for attempt in range(MAX_ENTRY_ORDER_RETRIES + 1):
            self.logger.info(f"Entry order attempt {attempt + 1} for {option_symbol}.")
            entry_order_id = self.order_manager.place_market_order_live(
                tradingsymbol=option_symbol,
                exchange=self.order_manager.kite.EXCHANGE_NFO, 
                transaction_type=self.order_manager.kite.TRANSACTION_TYPE_BUY,
                quantity=self.trade_quantity_actual,
                product=product_type 
            )
            if entry_order_id:
                self.logger.info(f"Entry order attempt {attempt + 1} successful. Order ID: {entry_order_id}")
                # Send Telegram message for successful order placement
                telegram_msg = (f"LiveTrader: ENTRY Order PLACED for {self.trade_quantity_actual} units of {option_symbol} ({self.option_type}).\n"
                                f"NIFTY Signal @ {nifty_signal_time.strftime('%H:%M:%S') if isinstance(nifty_signal_time, datetime) else nifty_signal_time} (Price: {nifty_price_at_signal:.2f}).\n"
                                f"Order ID: {entry_order_id}")
                self.order_manager.send_telegram_message(telegram_msg)
                break # Exit loop on success
            else:
                self.logger.warning(f"Entry order attempt {attempt + 1} failed for {option_symbol}.")
                if attempt < MAX_ENTRY_ORDER_RETRIES:
                    self.logger.info(f"Retrying in {ENTRY_ORDER_RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(ENTRY_ORDER_RETRY_DELAY_SECONDS)
                else:
                    self.logger.error(f"All {MAX_ENTRY_ORDER_RETRIES + 1} entry order attempts failed for {option_symbol}. No trade initiated.")
                    # Ensure entry_order_id remains None if all attempts fail
                    entry_order_id = None # Explicitly set to None
                    # Send Telegram message for entry order failure after all retries
                    telegram_msg = (f"LiveTrader: ALL FAILED Entry Order Attempts for {self.trade_quantity_actual} units of {option_symbol} ({self.option_type}).\n"
                                    f"NIFTY Signal @ {nifty_signal_time.strftime('%H:%M:%S') if isinstance(nifty_signal_time, datetime) else nifty_signal_time} (Price: {nifty_price_at_signal:.2f}). No trade initiated.")
                    self.order_manager.send_telegram_message(telegram_msg)
                    break # Exit loop after all retries failed

        if entry_order_id: # Proceed only if an order ID was obtained
            self.is_trade_active = True
            self.active_trade_details = {
                'option_symbol': option_symbol,
                'option_token': option_token,
                'option_lot_size': self.option_lot_size,
                'quantity_actual': self.trade_quantity_actual,
                'kite_entry_order_id': entry_order_id,
                'status': 'PENDING_ENTRY_CONFIRMATION',
                'nifty_signal_time': nifty_signal_time,
                'nifty_price_at_signal': nifty_price_at_signal,
                'max_hold_exit_time': datetime.now() + timedelta(minutes=self.max_holding_period_minutes)
            }
            self.logger.info(f"New {self.option_type} trade initiated. Trade details: {self.active_trade_details}")

    def _monitor_active_trade(self, latest_nifty_signal: int, latest_nifty_ohlc: pd.Series):
        if not self.active_trade_details or not self.is_trade_active:
            return

        status = self.active_trade_details.get('status')
        self.logger.debug(f"Monitoring active trade. Option: {self.active_trade_details.get('option_symbol')}, Status: {status}")

        if status == 'PENDING_ENTRY_CONFIRMATION':
            order_id = self.active_trade_details['kite_entry_order_id']
            self.logger.info(f"Checking entry order confirmation for {order_id}...")
            order_history = self.order_manager.get_order_history_live(order_id)
            avg_price = 0
            filled_quantity = 0
            order_status_from_history = 'UNKNOWN' # Default

            if order_history:
                for update in order_history:
                    if update['status'] == self.order_manager.kite.STATUS_COMPLETE:
                        order_status_from_history = self.order_manager.kite.STATUS_COMPLETE
                        trades_for_order = self.order_manager.get_trades_for_order_live(order_id)
                        if trades_for_order:
                            total_value = sum(trade['average_price'] * trade['quantity'] for trade in trades_for_order)
                            filled_quantity = sum(trade['quantity'] for trade in trades_for_order)
                            if filled_quantity > 0:
                                avg_price = total_value / filled_quantity
                            self.logger.info(f"Entry Order {order_id} trades: {trades_for_order}")
                        else: 
                             avg_price = update.get('average_price', 0) 
                        break 
                    elif update['status'] in [self.order_manager.kite.STATUS_REJECTED, self.order_manager.kite.STATUS_CANCELLED]:
                        order_status_from_history = update['status']
                        self.logger.error(f"Entry order {order_id} {order_status_from_history}. Reason: {update.get('status_message')}")
                        self._clear_active_trade_details(exit_reason='ENTRY_FAILED', exit_price=None)
                        return
            else:
                self.logger.warning(f"Could not fetch order history for entry order {order_id}. Will retry.")
                return 

            if order_status_from_history == self.order_manager.kite.STATUS_COMPLETE and avg_price > 0 and filled_quantity == self.active_trade_details['quantity_actual']:
                self.active_trade_details['entry_price_actual'] = avg_price
                self.active_trade_details['entry_timestamp'] = datetime.now() 
                self.active_trade_details['sl_price_calculated'] = avg_price * (1 - self.stop_loss_pct)
                self.active_trade_details['target_price_calculated'] = avg_price * (1 + self.profit_target_pct)
                self.active_trade_details['status'] = 'ACTIVE'
                self.logger.info(f"Trade Entry Confirmed for {self.active_trade_details['option_symbol']} @ {avg_price:.2f}. Qty: {filled_quantity}.\
                                 SL: {self.active_trade_details['sl_price_calculated']:.2f}, \
                                 TP: {self.active_trade_details['target_price_calculated']:.2f}, \
                                 Order ID: {order_id}")
                # Send Telegram message for entry confirmation
                telegram_msg = (f"LiveTrader: Trade Entry CONFIRMED for {self.active_trade_details['option_symbol']} @ {avg_price:.2f} (Qty: {filled_quantity}).\n"
                                f"SL: {self.active_trade_details['sl_price_calculated']:.2f}, TP: {self.active_trade_details['target_price_calculated']:.2f}.\n"
                                f"Entry Order ID: {order_id}")
                self.order_manager.send_telegram_message(telegram_msg)
            elif order_status_from_history == self.order_manager.kite.STATUS_COMPLETE: 
                 self.logger.error(f"Entry order {order_id} COMPLETE but avg_price ({avg_price}) or filled_qty ({filled_quantity}) is invalid. Required qty: {self.active_trade_details['quantity_actual']}. Treating as entry failure.")
                 self._clear_active_trade_details(exit_reason='ENTRY_CONFIRMATION_ERROR', exit_price=None)
            else:
                self.logger.info(f"Entry order {order_id} status is {order_status_from_history}. Waiting for completion.")

        elif status == 'ACTIVE':
            current_option_candle = self._get_last_option_candle(self.active_trade_details['option_token'])
            exit_trigger_reason = None
            exit_price_for_trigger = None 

            if not current_option_candle is None:
                self.logger.debug(f"Active Trade: Option {self.active_trade_details['option_symbol']} Candle Low: {current_option_candle['low']}, High: {current_option_candle['high']}")
                if current_option_candle['low'] <= self.active_trade_details['sl_price_calculated']:
                    exit_trigger_reason = 'STOP_LOSS_HIT'
                    exit_price_for_trigger = self.active_trade_details['sl_price_calculated'] 
                elif current_option_candle['high'] >= self.active_trade_details['target_price_calculated']:
                    exit_trigger_reason = 'TARGET_PRICE_HIT'
                    exit_price_for_trigger = self.active_trade_details['target_price_calculated'] 
            
            if not exit_trigger_reason and latest_nifty_signal == -1:
                exit_trigger_reason = 'STRATEGY_EXIT_SIGNAL'
            
            if not exit_trigger_reason and datetime.now() >= self.active_trade_details['max_hold_exit_time']:
                exit_trigger_reason = 'MAX_HOLD_TIME_REACHED'

            if exit_trigger_reason:
                self.logger.info(f"{exit_trigger_reason} condition met for option {self.active_trade_details['option_symbol']}. Attempting to exit.")
                product_type = self.config.get('LIVE_TRADER_SETTINGS', 'product_type', fallback=self.order_manager.kite.PRODUCT_MIS)
                exit_order_id = self.order_manager.place_market_order_live(
                    tradingsymbol=self.active_trade_details['option_symbol'],
                    exchange=self.order_manager.kite.EXCHANGE_NFO,
                    transaction_type=self.order_manager.kite.TRANSACTION_TYPE_SELL,
                    quantity=self.active_trade_details['quantity_actual'],
                    product=product_type
                )
                if exit_order_id:
                    self.active_trade_details['kite_exit_order_id'] = exit_order_id
                    self.active_trade_details['status'] = 'PENDING_EXIT_CONFIRMATION'
                    self.active_trade_details['exit_trigger_reason'] = exit_trigger_reason
                    self.active_trade_details['exit_price_for_trigger'] = exit_price_for_trigger 
                    self.logger.info(f"Exit order placed for {self.active_trade_details['option_symbol']}. Order ID: {exit_order_id}. Reason: {exit_trigger_reason}. Status: PENDING_EXIT_CONFIRMATION")
                    # Send Telegram message for successful exit order placement
                    telegram_msg = (f"LiveTrader: EXIT Order PLACED for {self.active_trade_details['option_symbol']}.\n"
                                    f"Reason: {exit_trigger_reason}. Order ID: {exit_order_id}.")
                    self.order_manager.send_telegram_message(telegram_msg)
                else:
                    self.logger.critical(f"CRITICAL: Failed to place EXIT order for {self.active_trade_details.get('option_symbol', 'UNKNOWN_SYMBOL')}! Position may be live. Terminating script.")
                    # Also send a telegram message if possible before exiting (though script might terminate too fast)
                    telegram_msg = (f"LiveTrader CRITICAL: FAILED to place EXIT order for {self.active_trade_details.get('option_symbol', 'UNKNOWN_SYMBOL')}! Position may be live.")
                    self.order_manager.send_telegram_message(telegram_msg)
                    sys.exit(f"CRITICAL_EXIT: EXIT_ORDER_PLACEMENT_FAILED for {self.active_trade_details.get('option_symbol', 'UNKNOWN_SYMBOL')}")
            else:
                self.logger.debug(f"No exit condition met for active trade {self.active_trade_details['option_symbol']}. Holding.")

        elif status == 'PENDING_EXIT_CONFIRMATION':
            order_id = self.active_trade_details['kite_exit_order_id']
            self.logger.info(f"Checking exit order confirmation for {order_id}...")
            order_history = self.order_manager.get_order_history_live(order_id)
            avg_price = 0
            filled_quantity = 0
            order_status_from_history = 'UNKNOWN'

            if order_history:
                for update in order_history:
                    if update['status'] == self.order_manager.kite.STATUS_COMPLETE:
                        order_status_from_history = self.order_manager.kite.STATUS_COMPLETE
                        trades_for_order = self.order_manager.get_trades_for_order_live(order_id)
                        if trades_for_order:
                            total_value = sum(trade['average_price'] * trade['quantity'] for trade in trades_for_order)
                            filled_quantity = sum(trade['quantity'] for trade in trades_for_order)
                            if filled_quantity > 0:
                                avg_price = total_value / filled_quantity
                            self.logger.info(f"Exit Order {order_id} trades: {trades_for_order}")
                        break
                    elif update['status'] in [self.order_manager.kite.STATUS_REJECTED, self.order_manager.kite.STATUS_CANCELLED]: 
                        order_status_from_history = update['status']
                        self.logger.error(f"Exit order {order_id} {order_status_from_history}! Reason: {update.get('status_message')}. This is unexpected for a market exit.")
                        self._clear_active_trade_details(exit_reason=self.active_trade_details.get('exit_trigger_reason', 'EXIT_CONFIRMATION_ERROR'), exit_price=None) 
                        return
            else:
                self.logger.warning(f"Could not fetch order history for exit order {order_id}. Will retry.")
                return

            if order_status_from_history == self.order_manager.kite.STATUS_COMPLETE and avg_price > 0 and filled_quantity == self.active_trade_details['quantity_actual']:
                exit_reason = self.active_trade_details.get('exit_trigger_reason', 'UNKNOWN_EXIT')
                actual_exit_price = avg_price 
                if exit_reason in ['STOP_LOSS_HIT', 'TARGET_PRICE_HIT'] and self.active_trade_details.get('exit_price_for_trigger') is not None:
                    self.logger.info(f"Exit for {exit_reason} was targetted at {self.active_trade_details['exit_price_for_trigger']:.2f}, actual fill at {avg_price:.2f}")
                
                self._clear_active_trade_details(exit_reason=exit_reason, exit_price=actual_exit_price)
            elif order_status_from_history == self.order_manager.kite.STATUS_COMPLETE: 
                self.logger.error(f"Exit order {order_id} COMPLETE but avg_price ({avg_price}) or filled_qty ({filled_quantity}) is invalid. Required qty: {self.active_trade_details['quantity_actual']}. Position might still be open! Manual check needed.")
                self._clear_active_trade_details(exit_reason=self.active_trade_details.get('exit_trigger_reason', 'EXIT_CONFIRMATION_ERROR_QTY'), exit_price=None)
            else:
                self.logger.info(f"Exit order {order_id} status is {order_status_from_history}. Waiting for completion.")

    def _clear_active_trade_details(self, exit_reason: str, exit_price: float | None):
        if not self.active_trade_details: return
        
        trade_summary = (f"--- Trade Closed ---\n"
                         f"Reason: {exit_reason}\n"
                         f"Option: {self.active_trade_details.get('option_symbol')}\n"
                         f"NIFTY @ Entry: {self.active_trade_details.get('nifty_signal_time')} (Price: {self.active_trade_details.get('nifty_price_at_signal')})\n"
                         f"Entry Order: {self.active_trade_details.get('kite_entry_order_id')}, Price: {self.active_trade_details.get('entry_price_actual')}, Time: {self.active_trade_details.get('entry_timestamp')}\n"
                         f"Exit Order: {self.active_trade_details.get('kite_exit_order_id', 'N/A')}, Price: {exit_price}, Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                         f"Qty: {self.active_trade_details.get('quantity_actual')} (Lots: {self.active_trade_details.get('quantity_actual',0) / self.active_trade_details.get('option_lot_size',1) if self.active_trade_details.get('option_lot_size') else 'N/A'})\n"
                         f"Calc SL: {self.active_trade_details.get('sl_price_calculated')}, Calc TP: {self.active_trade_details.get('target_price_calculated')}")

        if self.active_trade_details.get('entry_price_actual') is not None and exit_price is not None:
            pnl_per_unit = exit_price - self.active_trade_details['entry_price_actual']
            total_pnl = pnl_per_unit * self.active_trade_details['quantity_actual']
            pnl_summary = f"PNL/Unit: {pnl_per_unit:.2f}, Total PNL: {total_pnl:.2f}"
            trade_summary = f"{trade_summary}\n{pnl_summary}"
            self.logger.info(pnl_summary)
        else:
            trade_summary += "\nPNL not calculated (missing entry/exit price)."
            self.logger.info("PNL not calculated due to missing entry/exit price.")

        self.logger.info(trade_summary.replace('\n', ' | ')) # Log condensed summary
        self.order_manager.send_telegram_message(f"LiveTrader: {trade_summary}") # Send detailed summary to Telegram

        self.is_trade_active = False
        self.active_trade_details = {}
        self.logger.info(f"Active trade details cleared. Reason: {exit_reason}, Exit Price: {exit_price}")

    def _get_last_option_candle(self, option_token):
        try:
            today_str = date.today().strftime('%Y-%m-%d')
            # Fetch data for the last few minutes to ensure we get the latest completed candle
            # Kite historical data might have a slight delay for the absolute latest tick.
            # Fetching 2 candles and taking the second to last [-2] or last [-1] depending on timing.
            # For live monitoring, it's often best to get a slightly larger window if issues arise.
            from_datetime = datetime.now() - timedelta(minutes=5) # Fetch last 5 mins of data
            to_datetime = datetime.now()

            # Ensure from_date and to_date are strings in 'YYYY-MM-DD HH:MM:SS' format if API expects that,
            # or use date objects if API expects that. kite.historical_data uses date strings for from_date/to_date
            # and datetime objects for specific time ranges if supported.
            # For minute data for "today", from_date and to_date are usually the same date.

            # The Kite API historical_data takes 'from_date' and 'to_date' as YYYY-MM-DD strings.
            # It doesn't directly support fetching a time range within a day for 'minute' data across different days.
            # So, we fetch for 'today' and then filter by time if needed, or rely on getting the last few candles.

            option_data = self.order_manager.kite.historical_data(
                instrument_token=option_token,
                from_date=today_str, # Current day
                to_date=today_str,   # Current day
                interval='minute' 
            )

            if option_data and isinstance(option_data, list) and len(option_data) > 0:
                # The last element in the list is the most recent candle
                last_candle = option_data[-1]
                # Ensure it's a dictionary and has the required keys
                if isinstance(last_candle, dict) and 'low' in last_candle and 'high' in last_candle and 'date' in last_candle:
                    # Log the timestamp of the candle being used
                    self.logger.debug(f"Last option candle for token {option_token} at {last_candle['date']}: Low={last_candle['low']}, High={last_candle['high']}")
                    return {'low': last_candle['low'], 'high': last_candle['high'], 'date': last_candle['date']}
                else:
                    self.logger.warning(f"Last candle for {option_token} is not in expected format: {last_candle}")
                    return None
            else:
                self.logger.warning(f"No historical data returned for option token {option_token} for today.")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching last option candle for token {option_token}: {e}", exc_info=True)
            return None


# --- Main Execution --- #
if __name__ == '__main__':
    # Determine the path to the config file.
    # Assumes trading_config.ini is in the same directory or one level up from this script.
    _current_dir = os.path.dirname(__file__)
    _config_file = os.path.join(_current_dir, 'trading_config.ini')
    if not os.path.exists(_config_file):
        _config_file_alt = os.path.join(os.path.dirname(_current_dir), 'trading_config.ini')
        if os.path.exists(_config_file_alt):
            _config_file = _config_file_alt
        else:
            print(f"FATAL: trading_config.ini not found at {_config_file} or parent directory. Exiting.")
            exit(1)
    
    print(f"Using configuration file: {_config_file}")
    
    try:
        trader = LiveTrader(config_file_path=_config_file)
        trader.run_live_session()
    except FileNotFoundError as e:
        print(f"FATAL: Could not start LiveTrader due to missing file: {e}")
    except ValueError as e:
        print(f"FATAL: Could not start LiveTrader due to configuration error: {e}")
    except SystemExit as e:
        print(f"STOPPING: {e}")
    except Exception as e:
        print(f"FATAL: An unexpected error prevented LiveTrader from starting or running: {e}")
        # Detailed traceback can be logged by the logger inside the class if it reaches there
        # If error is before logger setup, print traceback here
        import traceback
        traceback.print_exc() 