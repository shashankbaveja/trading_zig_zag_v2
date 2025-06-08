import os
import sys
import logging
import traceback
import argparse
from datetime import datetime

from trader.live_trader import LiveTrader

def setup_logging():
    """Configures logging for the application."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, "live_trader_main.log")
    
    # Basic configuration to file and console
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_name, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Silence overly verbose library loggers
    logging.getLogger("mysql.connector").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

    logging.info("--- Main application logging configured ---")

def main():
    """
    The main entry point for the live trading application.
    Supports live mode and replay mode.
    
    To run in live mode: python main_live.py
    To run in replay mode for a specific timestamp: python main_live.py --replay "YYYY-MM-DD HH:MM:SS"
    """
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Live Trading Application")
    parser.add_argument('--replay', type=str, help='Run in replay mode from a specific timestamp (e.g., "YYYY-MM-DD HH:MM:SS")')
    args = parser.parse_args()

    config_file = 'config/trading_config.ini'
    
    if not os.path.exists(config_file):
        logging.critical(f"FATAL: trading_config.ini not found at {config_file}. Exiting.")
        sys.exit(1)
        
    logging.info(f"Using configuration file: {config_file}")
    
    replay_timestamp = None
    if args.replay:
        try:
            replay_timestamp = datetime.strptime(args.replay, '%Y-%m-%d %H:%M:%S')
            logging.warning(f"--- LAUNCHING IN REPLAY MODE FROM TIMESTAMP: {replay_timestamp} ---")
        except ValueError:
            logging.critical("Invalid replay timestamp format. Please use 'YYYY-MM-DD HH:MM:SS'. Exiting.")
            sys.exit(1)
    else:
        logging.info("--- LAUNCHING IN LIVE TRADING MODE ---")
    
    try:
        # Instantiate and run the live trader
        trader = LiveTrader(config_file_path=config_file, replay_timestamp=replay_timestamp)
        trader.run_session()
        
    except FileNotFoundError as e:
        logging.critical(f"FATAL: Could not start LiveTrader due to a missing file: {e}")
    except (ValueError, KeyError) as e:
        logging.critical(f"FATAL: Could not start LiveTrader due to a configuration error: {e}")
    except SystemExit as e:
        logging.info(f"STOPPING: {e}")
    except Exception as e:
        logging.critical(f"FATAL: An unexpected error occurred: {e}")
        logging.critical(traceback.format_exc())
        
    logging.info("--- Application has shut down ---")

if __name__ == '__main__':
    main() 