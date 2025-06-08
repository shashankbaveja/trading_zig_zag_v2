from datetime import datetime, time as dt_time
import logging
from trader.order_manager import OrderManager # Corrected import path

class SessionManager:
    """
    Manages the trading session with a simplified single operational window.
    """
    def __init__(self, config: dict, order_manager: OrderManager):
        """
        Initializes the SessionManager.
        Args:
            config (dict): A dictionary of live trader settings.
            order_manager (OrderManager): An instance for API access.
        """
        self.logger = logging.getLogger(__name__)
        self.order_manager = order_manager
        
        # Load the start and end times for the single operational window
        self.start_time = dt_time.fromisoformat(config.get('health_check_start_time', '09:14:00'))
        self.end_time = dt_time.fromisoformat(config.get('trading_end_time', '15:30:00'))
        
        # State variables
        self.is_session_active = True
        self.is_trade_allowed = False
        self.is_system_healthy = False
        
        self.logger.info(f"SessionManager initialized. Operational Window: {self.start_time} - {self.end_time}")

    def manage_session(self):
        """
        Checks if the current time is within the operational window and manages system health.
        """
        current_time = datetime.now().time()
        
        if not self.is_session_active:
            return

        # Check if we are within the single, continuous operational window
        if self.start_time <= current_time < self.end_time:
            # If system isn't healthy yet, run the check.
            if not self.is_system_healthy:
                self._run_health_check()
            
            # Trade permission is directly tied to system health during the window.
            self.is_trade_allowed = self.is_system_healthy
            
            if self.is_trade_allowed:
                self.logger.debug("Trading session active and system healthy.")
            else:
                self.logger.warning("Trading session active but system is NOT healthy. No trades will be executed.")
                
        else:
            # Outside the operational window
            self.is_trade_allowed = False
            if current_time >= self.end_time and self.is_session_active:
                self.logger.info("Trading session has ended for the day.")
                self.is_session_active = False

    def _run_health_check(self):
        """
        Performs a system health check by testing API connectivity.
        """
        self.logger.info("Performing system health check...")
        try:
            # Delegate the health check to the OrderManager
            if self.order_manager.check_api_connection():
                self.is_system_healthy = True
                self.logger.info("System health check PASSED. Kite API is responsive.")
            else:
                self.is_system_healthy = False
                self.logger.warning("System health check FAILED. Check OrderManager logs for details.")
        except Exception as e:
            self.is_system_healthy = False
            self.logger.error(f"System health check FAILED with exception: {e}", exc_info=True) 