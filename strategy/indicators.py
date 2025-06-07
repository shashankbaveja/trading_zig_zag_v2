import pandas as pd
import numpy as np
from collections import deque

def calculate_zigzag_pivots(data: pd.DataFrame, recent_pivots_deque: deque) -> list:
    """
    Calculates ZigZag pivot points (price and timestamp).
    Attempts to replicate the PineScript logic for `sz`.

    Args:
        data (pd.DataFrame): DataFrame with OHLC data and a DatetimeIndex.
        recent_pivots_deque (deque): A deque to store recent pivot log messages.

    Returns:
        list: A list of pivot point dictionaries: [{'timestamp': ts, 'price': price}, ...]
    """
    n = len(data)
    if n < 2:
        return []

    sz_points_values = pd.Series([np.nan] * n, index=data.index)
    pine_direction = 0.0  # 0: undetermined, 1: up, -1: down

    for i in range(1, n):
        current_ts = data.index[i]
        open_curr, close_curr = data['open'].iloc[i], data['close'].iloc[i]
        high_curr, low_curr = data['high'].iloc[i], data['low'].iloc[i]
        
        open_prev, close_prev = data['open'].iloc[i-1], data['close'].iloc[i-1]
        high_prev, low_prev = data['high'].iloc[i-1], data['low'].iloc[i-1]

        isUp_prev = close_prev >= open_prev
        isDown_prev = close_prev <= open_prev
        
        isUp_curr = close_curr >= open_curr
        isDown_curr = close_curr <= open_curr

        prev_pine_direction_for_calc = pine_direction
        
        if isUp_prev and isDown_curr:
            pine_direction = -1
        elif isDown_prev and isUp_curr:
            pine_direction = 1

        if isUp_prev and isDown_curr and prev_pine_direction_for_calc != -1:
            pivot_price = max(high_curr, high_prev)
            sz_points_values.iloc[i] = pivot_price
            recent_pivots_deque.append(f"PIVOT DETECTED: HIGH at {current_ts} - Price: {pivot_price:.2f}")
        elif isDown_prev and isUp_curr and prev_pine_direction_for_calc != 1:
            pivot_price = min(low_curr, low_prev)
            sz_points_values.iloc[i] = pivot_price
            recent_pivots_deque.append(f"PIVOT DETECTED: LOW at {current_ts} - Price: {pivot_price:.2f}")
            
    actual_pivots = []
    for idx_val, price in sz_points_values.items():
        if not pd.isna(price):
            actual_pivots.append({'timestamp': idx_val, 'price': price})
            
    return actual_pivots 