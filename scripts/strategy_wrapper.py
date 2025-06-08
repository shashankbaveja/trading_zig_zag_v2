import pandas as pd
from datetime import datetime, date
import configparser
import os

from trading_strategies import DataPrep, TradingStrategy, calculate_performance_metrics

def run_strategy_wrapper():
    """
    Wrapper to run the TradingStrategy with multiple configurations for 
    primary candlestick interval and alternate timeframe for ZigZag calculation.
    """
    print("--- Starting Strategy Wrapper --- ")

    # --- Configuration --- 
    config = configparser.ConfigParser()
    config_file_path = 'trading_config.ini'
    base_strategy_params = {}
    simulator_settings = {}

    if os.path.exists(config_file_path):
        config.read(config_file_path)
        if 'TRADING_STRATEGY' in config:
            base_strategy_params = dict(config['TRADING_STRATEGY'])
            print(f"Loaded TRADING_STRATEGY params from {config_file_path}: {base_strategy_params}")
        else:
            print(f"Warning: [TRADING_STRATEGY] section not found in {config_file_path}.")

        if 'SIMULATOR_SETTINGS' in config:
            simulator_settings = dict(config['SIMULATOR_SETTINGS'])
            print(f"Loaded SIMULATOR_SETTINGS from {config_file_path}: {simulator_settings}")
        else:
            print(f"Warning: [SIMULATOR_SETTINGS] section not found in {config_file_path}.")
    else:
        print(f"Warning: {config_file_path} not found. Using default parameters where applicable.")

    # --- Simulation Parameters (from config or defaults) ---
    try:
        sim_start_date_str = simulator_settings.get('simulation_start_date', "2025-05-01")
        sim_end_date_str = simulator_settings.get('simulation_end_date', "2025-05-25")
        sim_start_date = datetime.strptime(sim_start_date_str, '%Y-%m-%d').date()
        sim_end_date = datetime.strptime(sim_end_date_str, '%Y-%m-%d').date()
    except ValueError as e:
        print(f"Error parsing simulation dates: {e}. Using hardcoded defaults.")
        sim_start_date = date(2025, 5, 1)
        sim_end_date = date(2025, 5, 25)

    # instrument_token will now be a list iterated over
    # instrument_token = int(simulator_settings.get('instrument_token_to_trade', 256265)) 
    initial_capital = float(simulator_settings.get('initial_capital', 100000))
    warm_up_days = int(base_strategy_params.get('warm_up_days_for_strategy', 10))

    # --- Iteration Configurations --- 
    instrument_tokens_to_test_str = simulator_settings.get('instrument_tokens_to_test', '256265') # Default to NIFTY 50 if not specified
    primary_intervals_to_test_str = base_strategy_params.get('primary_intervals_to_test', '5,15,60')
    alttf_multipliers_str = base_strategy_params.get('alttf_intervals_to_test', '12,24')
    
    instrument_tokens_to_test = [int(t.strip()) for t in instrument_tokens_to_test_str.split(',') if t.strip()]
    primary_intervals_to_test = [int(p.strip()) for p in primary_intervals_to_test_str.split(',') if p.strip()]
    alttf_multipliers_to_test = [int(m.strip()) for m in alttf_multipliers_str.split(',') if m.strip()]

    print(f"\nSimulation Period: {sim_start_date} to {sim_end_date}")
    print(f"Initial Capital: {initial_capital}, Warm-up Days: {warm_up_days}")
    print(f"Instrument Tokens to Test: {instrument_tokens_to_test}")
    print(f"Primary Intervals to Test (minutes): {primary_intervals_to_test}")
    print(f"AltTF Multipliers for ZigZag to Test: {alttf_multipliers_to_test}")

    all_performance_results = []
    dp = DataPrep()
    if not dp.k_apis:
        print("CRITICAL: kiteAPIs could not be initialized in DataPrep. Exiting wrapper.")
        return

    # --- Main Loop for Combinations --- 
    for current_instrument_token in instrument_tokens_to_test:
        print(f"\n===== Processing for Instrument Token: {current_instrument_token} ====")
        for primary_interval in primary_intervals_to_test:
            print(f"\n--- Processing for Primary Interval: {primary_interval} minutes (Token: {current_instrument_token}) ---")
            
            print(f"Fetching data for token {current_instrument_token}, primary interval {primary_interval} min...")
            data_dict = dp.fetch_and_prepare_data(
                instrument_token=current_instrument_token,
                start_date_obj=sim_start_date,
                end_date_obj=sim_end_date,
                primary_interval_minutes=primary_interval,
                warm_up_days=warm_up_days
            )

            if not data_dict or data_dict['main_interval_data'].empty:
                print(f"Failed to fetch/prepare data for primary interval {primary_interval}. Skipping...")
                continue

            print(f"Successfully fetched main_interval_data ({len(data_dict['main_interval_data'])} rows) and one_minute_data ({len(data_dict['one_minute_data'])} rows).")

            for alt_tf_multiplier in alttf_multipliers_to_test:
                effective_alt_tf_minutes = primary_interval * alt_tf_multiplier
                
                # Skip if effective alt TF is less than or equal to primary (doesn't make sense for upsampling logic)
                # or if effective alt TF is not greater than 1 (as alt TF is usually a higher timeframe)
                if effective_alt_tf_minutes <= primary_interval and primary_interval > 1 : # Avoid altTF <= primary if primary itself isn't 1-min
                    if effective_alt_tf_minutes == primary_interval:
                         print(f"  -- Skipping AltTF Multiplier: {alt_tf_multiplier} for Primary: {primary_interval} min as Effective AltTF ({effective_alt_tf_minutes} min) is same as Primary.")
                    else: # effective_alt_tf_minutes < primary_interval
                         print(f"  -- Skipping AltTF Multiplier: {alt_tf_multiplier} for Primary: {primary_interval} min as Effective AltTF ({effective_alt_tf_minutes} min) would be less than Primary.")
                    continue
                if effective_alt_tf_minutes <= 1 and primary_interval <=1 and alt_tf_multiplier <=1: # if primary is 1 min, multiplier must make altTF > 1
                     print(f"  -- Skipping AltTF Multiplier: {alt_tf_multiplier} for Primary: {primary_interval} min as Effective AltTF ({effective_alt_tf_minutes} min) must be > 1 min. (Token: {current_instrument_token})")
                     continue


                print(f"  -- Testing with Token: {current_instrument_token}, Primary: {primary_interval} min, AltTF Multiplier: {alt_tf_multiplier} => Effective AltTF for ZigZag: {effective_alt_tf_minutes} minutes --")

                current_run_params = base_strategy_params.copy()
                current_run_params['alttf_interval_minutes'] = str(effective_alt_tf_minutes) 
                current_run_params['instrument_token'] = str(current_instrument_token) # Pass current token to strategy params

                try:
                    strategy_instance = TradingStrategy(
                        kite_apis_instance=dp.k_apis,
                        simulation_actual_start_date=sim_start_date,
                        **current_run_params 
                    )

                    print(f"  Running generate_signals for Token: {current_instrument_token}, Primary: {primary_interval} min, AltTF: {effective_alt_tf_minutes} min...")
                    signals_df = strategy_instance.generate_signals(data_input_dict=data_dict)
                    
                    if signals_df.empty:
                        print("  generate_signals returned empty DataFrame. Skipping performance calculation.")
                        performance_summary = {
                            'instrument_token': current_instrument_token,
                            'primary_interval_min': primary_interval,
                            'alttf_multiplier': alt_tf_multiplier,
                            'effective_alttf_min': effective_alt_tf_minutes,
                            'info': 'No signals generated or error in signal generation'
                        }
                    else:
                        print("  Calculating performance metrics...")
                        performance_metrics = calculate_performance_metrics(
                            signals_df,
                            initial_capital=initial_capital,
                            price_column='close', # Assuming 'close' is used for PnL
                            annual_rfr=float(simulator_settings.get('annual_risk_free_rate', 0.02))
                        )
                        performance_summary = {
                            'instrument_token': current_instrument_token,
                            'primary_interval_min': primary_interval,
                            'alttf_multiplier': alt_tf_multiplier,
                            'effective_alttf_min': effective_alt_tf_minutes,
                            **performance_metrics
                        }
                    
                    all_performance_results.append(performance_summary)
                    
                    pnl_to_display = performance_summary.get('total_pnl', 'N/A')
                    if isinstance(pnl_to_display, float):
                        pnl_display_str = f"{pnl_to_display:.2f}"
                    else:
                        pnl_display_str = str(pnl_to_display)
                    
                    print(f"  Performance for Token: {current_instrument_token}, Primary: {primary_interval}, AltTF Multiplier: {alt_tf_multiplier} (Effective: {effective_alt_tf_minutes} min) -> Trades: {performance_summary.get('total_trades', 'N/A')}, PnL: {pnl_display_str}")

                except Exception as e:
                    print(f"ERROR during strategy run for Token: {current_instrument_token}, Primary: {primary_interval}, AltTF Multiplier: {alt_tf_multiplier} (Effective: {effective_alt_tf_minutes} min): {e}")
                    error_summary = {
                        'instrument_token': current_instrument_token,
                        'primary_interval_min': primary_interval,
                        'alttf_multiplier': alt_tf_multiplier,
                        'effective_alttf_min': effective_alt_tf_minutes,
                        'error': str(e)
                    }
                    all_performance_results.append(error_summary)

    # --- Results Management --- 
    if all_performance_results:
        results_df = pd.DataFrame(all_performance_results)
        output_filename = "multi_run_performance_summary_15.csv"
        try:
            results_df.to_csv(output_filename, index=False)
            print(f"\n--- All configurations tested. Performance summary saved to {output_filename} ---")
            print(results_df.head())
        except Exception as e:
            print(f"Error saving performance summary to CSV: {e}")
    else:
        print("\nNo performance results generated from any configuration.")

    print("\n--- Strategy Wrapper Finished ---")

if __name__ == '__main__':
    run_strategy_wrapper() 