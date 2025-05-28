import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, time
import os
import configparser

from trading_strategies import TradingStrategy, DataPrep
from myKiteLib import kiteAPIs

config = configparser.ConfigParser()
_config_file_path = os.path.join(os.path.dirname(__file__), 'trading_config.ini')
if not os.path.exists(_config_file_path):
    _config_file_path_alt = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trading_config.ini')
    if os.path.exists(_config_file_path_alt):
        _config_file_path = _config_file_path_alt
    else:
        raise FileNotFoundError(f"trading_config.ini not found at {_config_file_path} or its parent directory.")
config.read(_config_file_path)
print(f"TradingSimulator: Configuration file '{_config_file_path}' loaded.")

class TradingSimulator:
    def __init__(self, 
                 strategy_obj: TradingStrategy,
                 trade_params: dict,
                 initial_capital: float,
                 allow_concurrent_trades: bool,
                 instrument_token: int
                 ):

        self.strategy_name = type(strategy_obj).__name__
        self.trade_params = trade_params
        if 'units' not in self.trade_params:
            raise ValueError("trade_params dict is missing required key: 'units'")
        self.initial_capital = initial_capital
        self.allow_concurrent_trades = allow_concurrent_trades
        self.instrument_token = instrument_token

        self.is_trade_active = False 
        self.active_index_trade = {}
        self.executed_trades = []

        print(f"TradingSimulator initialized.")
        print(f"Will operate on pre-generated strategy output for Index Token: {self.instrument_token}.")
        print(f"Strategy Name (for context): {self.strategy_name}")
        print(f"Trade Units: {self.trade_params['units']}")
        print(f"Initial Capital for Metrics: {self.initial_capital}")
        print(f"Allow Concurrent Trades: {self.allow_concurrent_trades}")

    def run_simulation(self, strategy_output_df: pd.DataFrame):
        print("\\n--- Starting Trading Simulation (on pre-generated strategy output) ---")
        self.executed_trades = []
        self.is_trade_active = False
        self.active_index_trade = {}
        last_exit_signal_processed_time = None

        if strategy_output_df.empty:
            print("TradingSimulator: Input strategy_output_df is empty. Cannot run simulation.")
            return pd.DataFrame()

        if 'signal' not in strategy_output_df.columns:
            raise ValueError("Signal column not found in the input strategy_output_df.")
        if 'date' not in strategy_output_df.columns and not isinstance(strategy_output_df.index, pd.DatetimeIndex):
            raise ValueError("Input strategy_output_df must have a 'date' column or a DatetimeIndex.")
        if 'open' not in strategy_output_df.columns or 'close' not in strategy_output_df.columns:
            raise ValueError("Input strategy_output_df must have 'open' and 'close' columns.")

        # Ensure 'date' is a column if it's an index
        sim_df = strategy_output_df.copy()
        if isinstance(sim_df.index, pd.DatetimeIndex) and 'date' not in sim_df.columns:
            sim_df.reset_index(inplace=True)
        
        # Ensure 'date' column is datetime
        try:
            sim_df['date'] = pd.to_datetime(sim_df['date'])
        except Exception as e:
            raise ValueError(f"Could not convert 'date' column to datetime in strategy_output_df: {e}")

        # Shift data to allow entry/exit on the 'open' of the candle *after* the signal candle.
        sim_df['next_open'] = sim_df['open'].shift(-1)
        sim_df['next_date'] = sim_df['date'].shift(-1)

        print(f"Total signals in input DF: BUY (1): {len(sim_df[sim_df['signal'] == 1])}, SELL/EXIT (-1): {len(sim_df[sim_df['signal'] == -1])}")

        for _, row in sim_df.iterrows():
            current_time = row['date']
            current_signal = row['signal']
            price_at_signal_candle_close = row['close']
            
            trade_action_price = row.get('next_open')
            trade_action_time = row.get('next_date')

            if pd.isna(trade_action_price) or pd.isna(trade_action_time):
                if self.is_trade_active and current_signal != -1: # Force close at end of data
                    self.active_index_trade['exit_time'] = current_time
                    self.active_index_trade['exit_price'] = price_at_signal_candle_close
                    self.active_index_trade['exit_reason'] = "End of Data"
                    
                    pnl_per_unit = self.active_index_trade['exit_price'] - self.active_index_trade['entry_price']
            total_pnl = pnl_per_unit * self.trade_params['units']
            
            trade_log_entry = {
                        'instrument_token': self.instrument_token,
                        'signal_time': self.active_index_trade['signal_time'],
                        'price_at_signal': self.active_index_trade['price_at_signal'],
                        'entry_time': self.active_index_trade['entry_time'],
                        'entry_price': round(self.active_index_trade['entry_price'], 2),
                        'exit_time': self.active_index_trade['exit_time'],
                        'exit_price': round(self.active_index_trade['exit_price'], 2),
                        'exit_reason': self.active_index_trade['exit_reason'],
                'pnl_per_unit': round(pnl_per_unit, 2),
                'total_pnl': round(total_pnl, 2),
                'units': self.trade_params['units']
            }
                    self.executed_trades.append(trade_log_entry)
                    print(f"  SIM_LOG: Closing Trade (End of Data): Signal @ {trade_log_entry['signal_time']}, Entry @ {trade_log_entry['entry_price']:.2f} ({trade_log_entry['entry_time']}), Exit @ {trade_log_entry['exit_price']:.2f} ({trade_log_entry['exit_time']}), PNL: {trade_log_entry['total_pnl']:.2f}")
                    self.is_trade_active = False
                    self.active_index_trade = {}
                continue

            if not self.allow_concurrent_trades and self.is_trade_active:
                if last_exit_signal_processed_time and current_time < last_exit_signal_processed_time:
                     pass
                elif current_signal == 1:
                    print(f"  SIM_LOG: Skipping Index BUY signal at {current_time} (Price: {price_at_signal_candle_close:.2f}) because a trade is already active and concurrent trades are false.")
                continue

            if current_signal == 1:
                if not self.is_trade_active or self.allow_concurrent_trades: # Allow new trade if not active OR concurrency is enabled
                    # For concurrent trades, each new buy signal would ideally start a new trade object.
                    # Current self.active_index_trade supports one active trade.
                    # If self.allow_concurrent_trades is true, this logic would need enhancement
                    # to manage multiple active_index_trades. For now, it will overwrite if true.
                    # Sticking to the simpler model: if allow_concurrent_trades is false, only one trade.
                    # If allow_concurrent_trades is true, the current model will still only track ONE active_index_trade.
                    # This means it will effectively behave like non-concurrent for now unless enhanced.
                    if not self.is_trade_active : # Standard non-concurrent entry
                        self.active_index_trade = {
                            'signal_time': current_time,
                            'price_at_signal': price_at_signal_candle_close,
                            'entry_time': trade_action_time,
                            'entry_price': trade_action_price,
                            'exit_reason': None
                        }
                        self.is_trade_active = True
                        print(f"  SIM_LOG: Initiating Index Trade: Signal @ {current_time}, Entry at next open: {trade_action_price:.2f} ({trade_action_time})")
                    elif self.allow_concurrent_trades:
                         print(f"  SIM_LOG: WARNING - Concurrent trade signaled at {current_time}, but current simplified simulator only tracks one active trade. This new signal will be ignored if a trade is already active.")
                         # To truly support concurrent trades, a list of active trades would be needed.

            elif current_signal == -1:
                if self.is_trade_active:
                    self.active_index_trade['exit_time'] = trade_action_time
                    self.active_index_trade['exit_price'] = trade_action_price
                    self.active_index_trade['exit_reason'] = "Strategy Exit Signal"
                    
                    pnl_per_unit = self.active_index_trade['exit_price'] - self.active_index_trade['entry_price']
                    total_pnl = pnl_per_unit * self.trade_params['units']
                    
                    trade_log_entry = {
                        'instrument_token': self.instrument_token,
                        'signal_time': self.active_index_trade['signal_time'],
                        'price_at_signal': self.active_index_trade['price_at_signal'],
                        'entry_time': self.active_index_trade['entry_time'],
                        'entry_price': round(self.active_index_trade['entry_price'], 2),
                        'exit_time': self.active_index_trade['exit_time'],
                        'exit_price': round(self.active_index_trade['exit_price'], 2),
                        'exit_reason': self.active_index_trade['exit_reason'],
                        'pnl_per_unit': round(pnl_per_unit, 2),
                        'total_pnl': round(total_pnl, 2),
                        'units': self.trade_params['units']
                    }
                    self.executed_trades.append(trade_log_entry)
                    print(f"  SIM_LOG: Closing Index Trade: Signal @ {trade_log_entry['signal_time']}, Entry @ {trade_log_entry['entry_price']:.2f} ({trade_log_entry['entry_time']}), Exit at next open: {trade_log_entry['exit_price']:.2f} ({trade_log_entry['exit_time']}), PNL: {trade_log_entry['total_pnl']:.2f}")
                    
                    self.is_trade_active = False
                    self.active_index_trade = {}
                    last_exit_signal_processed_time = trade_action_time
                else:
                    print(f"  SIM_LOG: Received Index EXIT signal at {current_time} but no active trade to close.")
        
        print("\\n--- Simulation Run Complete (on pre-generated strategy output) ---")
        return pd.DataFrame(self.executed_trades)

    def calculate_performance_metrics(self, trades_df: pd.DataFrame):
        print("\\n--- Trading Performance Metrics (Index Trades from pre-generated signals) ---")
        metrics_summary_dict = {}

        if trades_df.empty:
            print("No trades were executed. Cannot calculate metrics.")
            metrics_summary_dict["message"] = "No trades were executed."
            return metrics_summary_dict, "No trades were executed."

        num_total_trades = len(trades_df)
        trades_df['pnl_per_unit'] = pd.to_numeric(trades_df['pnl_per_unit'], errors='coerce')
        trades_df['total_pnl'] = pd.to_numeric(trades_df['total_pnl'], errors='coerce')
        trades_df.dropna(subset=['total_pnl'], inplace=True)

        if trades_df.empty : 
            print("No valid PNL data in trades. Cannot calculate metrics.")
            metrics_summary_dict["message"] = "No valid PNL data in trades."
            return metrics_summary_dict, "No valid PNL data in trades."

        winning_trades = trades_df[trades_df['total_pnl'] > 0]
        losing_trades = trades_df[trades_df['total_pnl'] < 0]
        num_winning_trades = len(winning_trades)
        num_losing_trades = len(losing_trades)
        win_rate = (num_winning_trades / num_total_trades) * 100 if num_total_trades > 0 else 0
        total_profit_loss = trades_df['total_pnl'].sum()
        average_pnl_per_trade = trades_df['total_pnl'].mean() if num_total_trades > 0 else 0
        gross_profit = winning_trades['total_pnl'].sum()
        gross_loss = abs(losing_trades['total_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        trades_df['cumulative_pnl'] = trades_df['total_pnl'].cumsum()
        
        profit_factor_str = f"{profit_factor:.2f}" if profit_factor != np.inf else "inf"

        trades_df['peak_pnl'] = trades_df['cumulative_pnl'].cummax()
        trades_df['drawdown'] = trades_df['peak_pnl'] - trades_df['cumulative_pnl']
        max_drawdown_value = trades_df['drawdown'].max() if not trades_df['drawdown'].empty else 0.0
        
        metrics_summary_dict = {
            "Total Trades Executed": num_total_trades,
            "Winning Trades": num_winning_trades,
            "Losing Trades": num_losing_trades,
            "Win Rate (%)": f"{win_rate:.2f}",
            "Total Profit/Loss": f"{total_profit_loss:.2f}",
            "Average Profit/Loss per Trade": f"{average_pnl_per_trade:.2f}",
            "Gross Profit": f"{gross_profit:.2f}",
            "Gross Loss": f"{gross_loss:.2f}",
            "Profit Factor": profit_factor_str,
            "Maximum Drawdown (Based on PNL)": f"{max_drawdown_value:.2f}"
        }
        summary_str_lines = ["--- Trading Performance Metrics (Index Trades from pre-generated signals) ---"]
        for key, value in metrics_summary_dict.items():
            line = f"{key}: {value}"
            print(line)
            summary_str_lines.append(line)
        return metrics_summary_dict, "\\n".join(summary_str_lines)

    def save_results(self, trades_df: pd.DataFrame, metrics_summary_str: str):
        output_dir = "cursor_logs"
        simulation_trades_log_filename = "simulation_trades_INDEX_output.csv"
        simulation_summary_filename = "simulation_summary_INDEX.txt"

        os.makedirs(output_dir, exist_ok=True)
        if not trades_df.empty:
            trades_log_path = os.path.join(output_dir, simulation_trades_log_filename)
            trades_df.to_csv(trades_log_path, index=False)
            print(f"\\nDetailed index trades log saved to: {os.path.abspath(trades_log_path)}")
        else:
            print("No index trades to save in the log.")
        summary_file_path = os.path.join(output_dir, simulation_summary_filename)
        try:
            with open(summary_file_path, 'w') as f:
                f.write(metrics_summary_str)
            print(f"Index performance summary saved to: {os.path.abspath(summary_file_path)}")
        except IOError as e:
            print(f"Error saving index performance summary: {e}")

if __name__ == '__main__':
    print("--- Trading Simulator Test (Operating on pre-generated strategy output) ---")
    
    sim_settings = config['SIMULATOR_SETTINGS']
    INDEX_TOKEN = sim_settings.getint('index_token')
    INITIAL_CAPITAL = sim_settings.getfloat('initial_capital')
    ALLOW_CONCURRENT_TRADES = sim_settings.getboolean('concurrent_signal_trade', fallback=False)
    SIMULATION_START_DATE_STR = sim_settings.get('simulation_start_date')
    SIMULATION_END_DATE_STR = sim_settings.get('simulation_end_date')

    STRATEGY_CONFIG_SECTION = 'TRADING_STRATEGY'
    if not config.has_section(STRATEGY_CONFIG_SECTION):
        raise ValueError(f"Strategy configuration section '[{STRATEGY_CONFIG_SECTION}]' not found in trading_config.ini")
    
    print(f"Using strategy configuration: [{STRATEGY_CONFIG_SECTION}]")
    strategy_config = config[STRATEGY_CONFIG_SECTION]

    TRADE_INTERVAL = strategy_config.get('trade_interval', 'minute')

    print("\\n--- Preparing Data and Generating Strategy Signals ---")
    data_preparator = DataPrep()
    if not data_preparator.k_apis:
        raise ConnectionError("__main__: Failed to initialize DataPrep's kiteAPIs.")

    try:
        start_date_obj = datetime.strptime(SIMULATION_START_DATE_STR, '%Y-%m-%d').date()
        end_date_obj = datetime.strptime(SIMULATION_END_DATE_STR, '%Y-%m-%d').date()
    except ValueError as e:
        print(f"Error parsing simulation dates: {e}. Using current date as fallback.")
        start_date_obj = date.today()
        end_date_obj = date.today()

    warm_up_days_for_strategy = strategy_config.getint('warm_up_days_for_strategy', 60)
    
    index_data_for_strategy = data_preparator.fetch_and_prepare_data(
        instrument_token=INDEX_TOKEN,
        start_date_obj=start_date_obj,
        end_date_obj=end_date_obj,
        interval=TRADE_INTERVAL,
        warm_up_days=warm_up_days_for_strategy 
    )

    if index_data_for_strategy.empty:
        raise ValueError("Failed to fetch data for the strategy.")

    params_for_strategy_class = {
        k.lower(): v for k, v in strategy_config.items()
        if k.lower() not in [
            'strategy_class_name', 'trade_units', 'trade_interval', 'warm_up_days_for_strategy',
        ]
    }
    active_trading_strategy = TradingStrategy(
        kite_apis_instance=data_preparator.k_apis, 
        simulation_actual_start_date=start_date_obj,
        **params_for_strategy_class
    )
    
    if 'date' in index_data_for_strategy.columns and not isinstance(index_data_for_strategy.index, pd.DatetimeIndex):
        index_data_for_strategy.set_index('date', inplace=True)
    elif 'date' not in index_data_for_strategy.columns and not isinstance(index_data_for_strategy.index, pd.DatetimeIndex):
        raise ValueError("Data for strategy must have 'date' column or be DatetimeIndexed.")

    strategy_output_with_signals_df = active_trading_strategy.generate_signals(index_data_for_strategy.copy())
    print(f"Strategy signals generated. Shape: {strategy_output_with_signals_df.shape}")

    sim_trade_params = {'units': strategy_config.getint('trade_units')}
    
    simulator_engine = TradingSimulator(
        strategy_obj=active_trading_strategy, 
        trade_params=sim_trade_params,
        initial_capital=INITIAL_CAPITAL,
        allow_concurrent_trades=ALLOW_CONCURRENT_TRADES,
        instrument_token=INDEX_TOKEN 
    )

    try:
        final_executed_trades_df = simulator_engine.run_simulation(strategy_output_with_signals_df)
        
        if not final_executed_trades_df.empty:
            print("\\n--- Final Executed Index Trades --- (First 5)")
            print(final_executed_trades_df.head())
            metrics_dict, metrics_summary_str = simulator_engine.calculate_performance_metrics(final_executed_trades_df)
            simulator_engine.save_results(final_executed_trades_df, metrics_summary_str)
        else:
            print("Simulation completed with no index trades executed.")
            _, metrics_summary_str = simulator_engine.calculate_performance_metrics(pd.DataFrame())
            simulator_engine.save_results(pd.DataFrame(), metrics_summary_str)

    except Exception as e:
        print(f"An unexpected error occurred during simulation run: {e}")
        import traceback
        traceback.print_exc()

    print("\\n--- Trading Simulator Test Complete (Operating on pre-generated strategy output) ---") 