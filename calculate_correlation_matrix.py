import pandas as pd

def calculate_and_save_correlation(csv_path="index_data.csv", output_path="correlation_matrix.csv"):
    """
    Calculates the correlation matrix of price returns from OHLCV data.

    Args:
        csv_path (str): Path to the input CSV file.
                        Expected columns: 'timestamp', 'instrument_token', 'close'.
        output_path (str): Path to save the resulting correlation matrix CSV.
    """
    print(f"Starting correlation matrix calculation from {csv_path}...")

    try:
        # Load the dataset
        # For very large files, consider using chunksize in read_csv
        print("Loading data...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows.")

        # Ensure correct data types
        print("Converting data types...")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Assuming 'close' is already numeric, if not, add:
        # df['close'] = pd.to_numeric(df['close'], errors='coerce')
        # df['instrument_token'] = df['instrument_token'].astype(str) # Ensure instrument_token is string for pivoting

        # Pivot the table to get instrument tokens as columns and close prices as values
        print("Pivoting data...")
        pivot_df = df.pivot_table(index='timestamp', columns='instrument_token', values='close')
        print(f"Pivoted table shape: {pivot_df.shape}")

        # Calculate percentage returns
        # It's generally better to use returns for correlation analysis as price series are often non-stationary.
        print("Calculating percentage returns...")
        returns_df = pivot_df.pct_change()

        # Drop the first row of NaNs created by pct_change()
        returns_df = returns_df.dropna(how='all', axis=0)
        print(f"Returns table shape after dropping NaNs: {returns_df.shape}")

        if returns_df.empty or len(returns_df) < 2:
            print("Not enough data points after processing to calculate correlation.")
            print("This might be due to issues in pivoting or all NaNs in returns.")
            if not pivot_df.empty:
                print(f"Pivot table head:\n{pivot_df.head()}")
                # Check for common issues in pivot
                # e.g. if all close prices for an instrument are the same, pct_change will be 0 or NaN
                problematic_instruments = pivot_df.columns[pivot_df.nunique() <= 1].tolist()
                if problematic_instruments:
                    print(f"Instruments with single unique close values (potential issue for pct_change): {problematic_instruments}")
            return

        # Calculate the correlation matrix
        print("Calculating correlation matrix...")
        correlation_matrix = returns_df.corr(method='pearson')
        print(f"Correlation matrix shape: {correlation_matrix.shape}")

        # Save the correlation matrix
        print(f"Saving correlation matrix to {output_path}...")
        correlation_matrix.to_csv(output_path)
        print(f"Successfully saved correlation matrix to {output_path}")
        print("\nCorrelation Matrix Head:")
        print(correlation_matrix.head())

    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
    except KeyError as e:
        print(f"Error: A required column is missing from the CSV: {e}")
        print("Please ensure the CSV contains 'timestamp', 'instrument_token', and 'close' columns.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define the input and output file paths
    input_csv_file = "index_data.csv"
    output_csv_file = "index_correlation_matrix.csv" # Changed output filename for clarity

    calculate_and_save_correlation(csv_path=input_csv_file, output_path=output_csv_file) 