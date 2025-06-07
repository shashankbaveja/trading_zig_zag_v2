import logging
import time
from datetime import datetime
from myKiteLib import OrderPlacement as KiteOrderPlacement

class OrderManager:
    """
    Handles all interactions with the broker's API for order management.
    Acts as a wrapper around the myKiteLib.OrderPlacement class.
    """
    MAX_ENTRY_ORDER_RETRIES = 2
    ENTRY_ORDER_RETRY_DELAY_SECONDS = 3

    def __init__(self, config: dict):
        """
        Initializes the OrderManager.
        Args:
            config (dict): A dictionary of live trader settings.
        """
        self.logger = logging.getLogger(__name__)
        try:
            self.kite_om = KiteOrderPlacement()
            self.logger.info("Initializing Kite trading session...")
            self.kite_om.init_trading() # Handles token validation
            self.logger.info("Kite trading session initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kite Order Placement: {e}", exc_info=True)
            raise SystemExit("Core component initialization failed.")

        # Load trading parameters from config
        self.trading_token_symbol = config.get('trading_token_symbol', f"NIFTY{datetime.now().strftime('%y%b').upper()}FUT")
        self.exchange = self.kite_om.kite.EXCHANGE_NFO
        self.product_type = config.get('product_type', 'NRML')
        self.quantity = int(config.get('trade_quantity', 75))
        
        self.logger.info(f"OrderManager initialized for symbol: {self.trading_token_symbol}")

    def place_entry_order(self, direction: str, pattern_name: str) -> str | None:
        """
        Places a market order to enter a new trade.

        Args:
            direction (str): 'LONG' or 'SHORT'.
            pattern_name (str): The name of the pattern triggering the entry.

        Returns:
            The order ID if successful, otherwise None.
        """
        transaction_type = (self.kite_om.kite.TRANSACTION_TYPE_BUY if direction == 'LONG' 
                            else self.kite_om.kite.TRANSACTION_TYPE_SELL)
        
        self.logger.info(f"Attempting to place {direction} entry order for {pattern_name}.")

        for attempt in range(1, self.MAX_ENTRY_ORDER_RETRIES + 1):
            try:
                order_id = self.kite_om.place_market_order_live(
                    tradingsymbol=self.trading_token_symbol,
                    exchange=self.exchange,
                    transaction_type=transaction_type,
                    quantity=self.quantity,
                    product=self.product_type,
                    tag=pattern_name
                )
                if order_id:
                    self.logger.info(f"Entry order placed on attempt {attempt}. Order ID: {order_id}")
                    self.send_telegram_notification(f"🟢 ENTRY Order Placed 🟢\nDirection: {direction}\nPattern: {pattern_name}\nOrder ID: {order_id}")
                    return order_id
                
                self.logger.warning(f"Order placement returned None on attempt {attempt}.")
            except Exception as e:
                self.logger.error(f"Order placement attempt {attempt} failed: {e}", exc_info=True)
            
            if attempt < self.MAX_ENTRY_ORDER_RETRIES:
                time.sleep(self.ENTRY_ORDER_RETRY_DELAY_SECONDS)

        self.logger.error(f"Failed to place entry order after {self.MAX_ENTRY_ORDER_RETRIES} attempts.")
        self.send_telegram_notification(f"🚨 FAILED to place {direction} entry order for {pattern_name}!")
        return None

    def place_exit_order(self, direction: str, reason: str) -> str | None:
        """
        Places a market order to exit an active trade.

        Args:
            direction (str): The direction of the trade being exited ('LONG' or 'SHORT').
            reason (str): The reason for the exit.

        Returns:
            The exit order ID if successful, otherwise None.
        """
        exit_transaction_type = (self.kite_om.kite.TRANSACTION_TYPE_SELL if direction == 'LONG'
                                 else self.kite_om.kite.TRANSACTION_TYPE_BUY)
        
        try:
            order_id = self.kite_om.place_market_order_live(
                tradingsymbol=self.trading_token_symbol,
                exchange=self.exchange,
                transaction_type=exit_transaction_type,
                quantity=self.quantity,
                product=self.product_type,
                tag=reason
            )
            self.logger.info(f"Exit order placed for {direction} trade. Reason: {reason}. Order ID: {order_id}")
            self.send_telegram_notification(f"🔴 EXIT Order Placed 🔴\nDirection: {direction}\nReason: {reason}\nOrder ID: {order_id}")
            return order_id
        except Exception as e:
            self.logger.error(f"Failed to place exit order: {e}", exc_info=True)
            self.send_telegram_notification(f"🚨 FAILED to place exit order for {direction} trade!")
            return None

    def get_order_average_price(self, order_id: str) -> float | None:
        """
        Checks an order's history and returns the average fill price.
        """
        try:
            history = self.kite_om.get_order_history_live(order_id)
            if history:
                for update in reversed(history): # Check from the latest update
                    if update['status'] == self.kite_om.kite.STATUS_COMPLETE:
                        avg_price = float(update.get('average_price', 0.0))
                        if avg_price > 0:
                            self.logger.info(f"Order {order_id} confirmed filled at avg price: {avg_price}")
                            self.send_telegram_notification(f"✅ Order Confirmed ✅\nID: {order_id}\nAvg Price: {avg_price}")
                            return avg_price
            return None
        except Exception as e:
            self.logger.error(f"Error checking order confirmation for {order_id}: {e}")
            return None

    def send_telegram_notification(self, message: str):
        """Sends a message to Telegram."""
        try:
            self.kite_om.send_telegram_message(message)
        except Exception as e:
            self.logger.error(f"Failed to send Telegram notification: {e}") 