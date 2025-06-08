#!/bin/bash

# Live Trader Stop Script
# This script gracefully stops the live trader

# Set working directory
cd "/Users/shashankbaveja/Main/Projects/KiteConnectAPI/trading_setup ZigZag"

# Create logs directory if it doesn't exist
mkdir -p logs

# Log file
CRON_LOG_FILE="logs/cron_stop_$(date +%Y%m%d).log"

echo "$(date): Stopping Live Trader..." >> "$CRON_LOG_FILE"

# Method 1: Try to stop using saved PID
if [ -f /tmp/live_trader.pid ]; then
    PID=$(cat /tmp/live_trader.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "$(date): Sending SIGTERM to PID: $PID" >> "$CRON_LOG_FILE"
        kill -TERM $PID
        
        # Wait up to 10 seconds for graceful shutdown
        for i in {1..10}; do
            if ! ps -p $PID > /dev/null 2>&1; then
                echo "$(date): Live Trader stopped gracefully." >> "$CRON_LOG_FILE"
                rm -f /tmp/live_trader.pid
                exit 0
            fi
            sleep 1
        done
        
        # Force kill if still running
        echo "$(date): Force killing PID: $PID" >> "$CRON_LOG_FILE"
        kill -9 $PID
        rm -f /tmp/live_trader.pid
    else
        echo "$(date): PID $PID not found. Cleaning up PID file." >> "$CRON_LOG_FILE"
        rm -f /tmp/live_trader.pid
    fi
fi

# Method 2: Kill any remaining live_trader.py processes
if pgrep -f "python live_trader.py" > /dev/null; then
    echo "$(date): Killing remaining live_trader.py processes..." >> "$CRON_LOG_FILE"
    pkill -f "python live_trader.py"
    sleep 2
    
    # Force kill if still running
    if pgrep -f "python live_trader.py" > /dev/null; then
        echo "$(date): Force killing remaining processes..." >> "$CRON_LOG_FILE"
        pkill -9 -f "python live_trader.py"
    fi
fi

echo "$(date): Live Trader stop complete." >> "$CRON_LOG_FILE" 