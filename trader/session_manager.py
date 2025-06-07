from datetime import datetime, time as dt_time
import logging
from myKiteLib import OrderPlacement

class SessionManager:
    """
    Manages the trading session, including time windows and system health.
    """
    def __init__(self, config: dict, order_manager: OrderPlacement):
        """
        Initializes the SessionManager.

        Args:
            config (dict): A dictionary of live trader settings.
            order_manager (OrderPlacement): An instance for API access.
        """
        self.logger = logging.getLogger(__name__)
        self.order_manager = order_manager
        
        # Load timing settings from config
        self.trading_start_time = dt_time.fromisoformat(config.get('trading_start_time', '09:20:00'))
        self.trading_end_time = dt_time.fromisoformat(config.get('trading_end_time', '15:00:00'))
        self.health_check_start_time = dt_time.fromisoformat(config.get('health_check_start_time', '09:15:00'))
        self.health_check_end_time = dt_time.fromisoformat(config.get('health_check_end_time', '09:20:00'))
        
        self.signal_token = int(config.get('signal_token', 256265))
        
        # State variables
        self.is_session_active = True
        self.is_trade_allowed = False
        self.is_system_healthy = False
        
        self.logger.info(f"SessionManager initialized. Trading Hours: {self.trading_start_time}-{self.trading_end_time}")

    def manage_session(self):
        """
        Checks the current time and system health to determine session state.
        This should be called at the start of each trading loop iteration.
        """
        current_time = datetime.now().time()
        
        if not self.is_session_active:
            return

        if self.health_check_start_time <= current_time < self.health_check_end_time:
            if not self.is_system_healthy: # Only check if not already confirmed healthy
                self._run_health_check()
            self.is_trade_allowed = False
            self.logger.debug("In health check window. Trade execution disabled.")
            
        elif self.trading_start_time <= current_time < self.trading_end_time:
            if not self.is_system_healthy:
                self.logger.info("Trading window started but system not healthy. Re-checking...")
                self._run_health_check()
            
            self.is_trade_allowed = self.is_system_healthy
            if self.is_trade_allowed:
                self.logger.debug("Trading window active and system healthy.")
            else:
                self.logger.warning("Trading window active but system NOT healthy.")
                
        else:
            self.is_trade_allowed = False
            if current_time >= self.trading_end_time:
                if self.is_session_active:
                    self.logger.info("Trading session has ended for the day.")
                    self.is_session_active = False

    def _run_health_check(self):
        """
        Performs a system health check by testing API connectivity.
        """
        self.logger.info("Performing system health check...")
        try:
            # A simple API call to check connectivity
            profile = self.order_manager.kite.profile()
            if profile and isinstance(profile, dict) and 'user_id' in profile:
                self.is_system_healthy = True
                self.logger.info("System health check PASSED. Kite API is responsive.")
            else:
                self.is_system_healthy = False
                self.logger.warning("System health check FAILED. API returned unexpected response.")
        except Exception as e:
            self.is_system_healthy = False
            self.logger.error(f"System health check FAILED with exception: {e}", exc_info=True) 