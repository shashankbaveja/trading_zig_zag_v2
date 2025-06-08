import pandas as pd

def calculate_performance_metrics(signals_df: pd.DataFrame, 
                                  initial_capital: float, 
                                  price_column: str = 'close',
                                  **kwargs) -> dict:
    """
    Calculates PnL and other performance metrics from strategy signals.
    This function is designed for backtesting analysis.

    Args:
        signals_df: DataFrame from a backtest run, must include 'signal', 
                    'pattern_tag', and price_column. Index must be DatetimeIndex.
        initial_capital: The starting capital for the simulation.
        price_column: Column in signals_df to use for entry/exit prices.
        **kwargs: Can include 'annual_rfr' (e.g., 0.02 for 2%).

    Returns:
        A dictionary containing various performance metrics.
    """
    if not isinstance(signals_df.index, pd.DatetimeIndex):
        print("Error in calculate_performance_metrics: signals_df must have a DatetimeIndex.")
        return {}
    if price_column not in signals_df.columns:
        print(f"Error in calculate_performance_metrics: price_column '{price_column}' not found.")
        return {}
    if 'signal' not in signals_df.columns:
        print("Error in calculate_performance_metrics: 'signal' column not found.")
        return {}
    if 'pattern_tag' not in signals_df.columns:
        print("Error in calculate_performance_metrics: 'pattern_tag' column not found.")
        return {}

    active_long_trade = None
    active_short_trade = None
    completed_trades = []

    for current_dt, row in signals_df.iterrows():
        current_price = row[price_column]
        signal_value = row['signal']
        pattern_tag_text = str(row['pattern_tag']).lower()

        if signal_value == -1: # Exit signal
            if "long exit" in pattern_tag_text and active_long_trade:
                pnl = current_price - active_long_trade['entry_price']
                completed_trades.append({
                    'entry_time': active_long_trade['entry_time'], 'exit_time': current_dt,
                    'entry_price': active_long_trade['entry_price'], 'exit_price': current_price,
                    'trade_type': 'long', 'pnl': pnl
                })
                active_long_trade = None
            elif "short exit" in pattern_tag_text and active_short_trade:
                pnl = active_short_trade['entry_price'] - current_price
                completed_trades.append({
                    'entry_time': active_short_trade['entry_time'], 'exit_time': current_dt,
                    'entry_price': active_short_trade['entry_price'], 'exit_price': current_price,
                    'trade_type': 'short', 'pnl': pnl
                })
                active_short_trade = None
        
        if signal_value == 1: # Entry signal
            if "long entry" in pattern_tag_text and not active_long_trade:
                active_long_trade = {'entry_price': current_price, 'entry_time': current_dt}
            elif "short entry" in pattern_tag_text and not active_short_trade:
                active_short_trade = {'entry_price': current_price, 'entry_time': current_dt}

    metrics = {}
    if not completed_trades:
        metrics['total_trades'] = 0
        metrics['info'] = "No completed trades found to calculate metrics."
        return metrics

    trades_df = pd.DataFrame(completed_trades)
    trades_df.sort_values(by='exit_time', inplace=True)

    metrics['total_trades'] = len(trades_df)
    winning_trades_df = trades_df[trades_df['pnl'] > 0]
    losing_trades_df = trades_df[trades_df['pnl'] < 0]
    
    metrics['winning_trades'] = len(winning_trades_df)
    metrics['losing_trades'] = len(losing_trades_df)
    
    metrics['win_rate_pct'] = (metrics['winning_trades'] / metrics['total_trades']) * 100 if metrics['total_trades'] > 0 else 0
    metrics['total_pnl'] = trades_df['pnl'].sum()
    metrics['gross_profit'] = winning_trades_df['pnl'].sum()
    metrics['gross_loss'] = abs(losing_trades_df['pnl'].sum())

    metrics['average_pnl_per_trade'] = metrics['total_pnl'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
    metrics['average_profit_per_winning_trade'] = metrics['gross_profit'] / metrics['winning_trades'] if metrics['winning_trades'] > 0 else 0
    metrics['average_loss_per_losing_trade'] = metrics['gross_loss'] / metrics['losing_trades'] if metrics['losing_trades'] > 0 else 0

    if metrics['gross_loss'] > 0:
        metrics['profit_factor'] = metrics['gross_profit'] / metrics['gross_loss']
    elif metrics['gross_profit'] > 0:
        metrics['profit_factor'] = float('inf')
    else:
        metrics['profit_factor'] = 0

    # Max Drawdown
    trades_df['equity_curve'] = initial_capital + trades_df['pnl'].cumsum()
    trades_df['running_max_equity'] = trades_df['equity_curve'].cummax()
    trades_df['drawdown_absolute'] = trades_df['running_max_equity'] - trades_df['equity_curve']
    
    metrics['max_drawdown_absolute'] = trades_df['drawdown_absolute'].max()
    if pd.isna(metrics['max_drawdown_absolute']):
        metrics['max_drawdown_absolute'] = 0.0

    if metrics['max_drawdown_absolute'] > 0:
        idx_max_drawdown = trades_df['drawdown_absolute'].idxmax()
        peak_equity = trades_df.loc[idx_max_drawdown, 'running_max_equity']
        metrics['max_drawdown_percentage'] = (metrics['max_drawdown_absolute'] / peak_equity) * 100 if peak_equity > 0 else float('inf')
    else:
        metrics['max_drawdown_percentage'] = 0.0

    # Sharpe Ratio
    if len(trades_df) >= 2:
        trades_df['pnl_pct'] = trades_df.apply(
            lambda row: (row['pnl'] / row['entry_price']) if row['entry_price'] != 0 else 0.0, axis=1
        )
        trades_df['holding_period_days'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / (24 * 3600.0)
        annual_rfr = kwargs.get('annual_rfr', 0.02)
        trades_df['rfr_per_trade'] = (annual_rfr / 365.0) * trades_df['holding_period_days']
        trades_df['excess_return_pct'] = trades_df['pnl_pct'] - trades_df['rfr_per_trade']
        
        mean_excess_return = trades_df['excess_return_pct'].mean()
        std_excess_return = trades_df['excess_return_pct'].std()

        if pd.isna(std_excess_return) or std_excess_return < 1e-9:
            metrics['sharpe_ratio_per_trade'] = float('inf') if mean_excess_return > 1e-9 else 0.0
        else:
            metrics['sharpe_ratio_per_trade'] = mean_excess_return / std_excess_return
    else:
        metrics['sharpe_ratio_per_trade'] = 0.0

    return metrics 