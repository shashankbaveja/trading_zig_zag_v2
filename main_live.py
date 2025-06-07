import os
import sys
import logging
import traceback

from trader.live_trader import LiveTrader

def setup_logging():
    """Configures logging for the application."""
    log_file_name = "live_trader_main.log"
    # Basic configuration to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_name, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("--- Main application logging configured ---")

def main():
    """
    The main entry point for the live trading application.
    Supports live mode and replay mode.
    
    To run in live mode: python main_live.py
    To run in replay mode: python main_live.py YYYY-MM-DD
    """
    setup_logging()
    
    # Determine the config file path
    # This assumes the script is run from the project's root directory
    config_file = 'trading_config.ini'
    
    if not os.path.exists(config_file):
        logging.critical(f"FATAL: trading_config.ini not found at {config_file}. Exiting.")
        sys.exit(1)
        
    logging.info(f"Using configuration file: {config_file}")
    
    # Check for replay date argument
    replay_date_str = None
    if len(sys.argv) > 1:
        replay_date_str = sys.argv[1]
        logging.warning(f"--- LAUNCHING IN REPLAY MODE FOR DATE: {replay_date_str} ---")
    else:
        logging.info("--- LAUNCHING IN LIVE TRADING MODE ---")
    
    try:
        # Instantiate and run the live trader
        trader = LiveTrader(config_file_path=config_file, replay_date=replay_date_str)
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