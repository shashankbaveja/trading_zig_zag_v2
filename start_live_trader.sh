#!/bin/bash

# Live Trader Start Script
# This script activates the conda environment and starts the live trader

# Set working directory
cd "/Users/shashankbaveja/Main/Projects/KiteConnectAPI/trading_setup ZigZag"

# Create logs directory if it doesn't exist
mkdir -p logs

# Log file with date
LOG_FILE="logs/live_trader_$(date +%Y%m%d).log"
CRON_LOG_FILE="logs/cron_start_$(date +%Y%m%d).log"

echo "$(date): Starting Live Trader..." >> "$CRON_LOG_FILE"

# Check if the script is already running
if pgrep -f "python live_trader.py" > /dev/null; then
    echo "$(date): Live Trader is already running. Exiting." >> "$CRON_LOG_FILE"
    exit 1
fi

# Activate conda environment and run the script
source /opt/anaconda3/bin/activate
conda activate /opt/anaconda3/envs/KiteConnect

# Start the live trader in background with nohup
nohup python live_trader.py >> "$LOG_FILE" 2>&1 &

# Get the process ID
PID=$!
echo "$(date): Live Trader started with PID: $PID" >> "$CRON_LOG_FILE"

# Save PID to file for stopping later
echo $PID > /tmp/live_trader.pid

echo "$(date): Live Trader startup complete." >> "$CRON_LOG_FILE" 