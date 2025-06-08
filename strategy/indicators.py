import pandas as pd
import numpy as np
from collections import deque
from trader.myKiteLib import convert_minute_data_interval
import os
from datetime import datetime

def calculate_zigzag_pivots(data: pd.DataFrame, recent_pivots_deque: deque, resample_interval_minutes: int = 1) -> list:
    """
    Calculates ZigZag pivot points (price and timestamp).
    Attempts to replicate the PineScript logic for `sz`.

    Args:
        data (pd.DataFrame): DataFrame with OHLC data and a DatetimeIndex.
        recent_pivots_deque (deque): A deque to store recent pivot log messages.
        resample_interval_minutes (int): If > 1, resamples the data to this interval. Defaults to 1 (no resampling).

    Returns:
        list: A list of pivot point dictionaries: [{'timestamp': ts, 'price': price, 'type': 'HIGH'|'LOW'}, ...]
    """
    if resample_interval_minutes > 1:
        # Data from DataHandler has a DatetimeIndex. `convert_minute_data_interval` expects a 'timestamp' column.
        data_for_resample = data.reset_index().rename(columns={'index': 'timestamp'})
        
        # Resample the data
        resampled_data = convert_minute_data_interval(data_for_resample, to_interval=resample_interval_minutes)
        
        if resampled_data.empty:
            return []
        data = resampled_data.set_index('timestamp')


    n = len(data)
    if n < 2:
        return []

    sz_points_values = pd.Series([np.nan] * n, index=data.index)
    sz_points_types = pd.Series([''] * n, index=data.index) # Series to hold type
    pine_direction = 0.0

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
            sz_points_types.iloc[i] = 'HIGH' # Store type
            recent_pivots_deque.append(f"PIVOT DETECTED: HIGH at {current_ts} - Price: {pivot_price:.2f}")
        elif isDown_prev and isUp_curr and prev_pine_direction_for_calc != 1:
            pivot_price = min(low_curr, low_prev)
            sz_points_values.iloc[i] = pivot_price
            sz_points_types.iloc[i] = 'LOW' # Store type
            recent_pivots_deque.append(f"PIVOT DETECTED: LOW at {current_ts} - Price: {pivot_price:.2f}")
            
    actual_pivots = []
    for i in range(n):
        price = sz_points_values.iloc[i]
        pivot_type = sz_points_types.iloc[i]
        if not pd.isna(price):
            actual_pivots.append({'timestamp': sz_points_values.index[i], 'price': price, 'type': pivot_type})
            
    return actual_pivots 