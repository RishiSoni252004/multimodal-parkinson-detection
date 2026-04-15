#!/bin/bash

# Navigate to the correct directory just in case
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================="
echo " Shutting Down Parkinson's App System     "
echo "=========================================="

if [ -f app_pid.txt ]; then
    PID=$(cat app_pid.txt)
    
    # Check if process is actually running
    if ps -p $PID > /dev/null; then
        echo "Stopping main Streamlit process (PID $PID)..."
        kill -15 $PID
        sleep 2
        
        # Verify it really shut down
        if ps -p $PID > /dev/null; then
             echo "Force terminating process..."
             kill -9 $PID
        fi
        
        echo "✅ Application successfully shut down."
    else
        echo "Process $PID was already stopped."
    fi
    
    # Clean up the stale PID file
    rm app_pid.txt
else
    echo "⚠️ app_pid.txt not found. Attempting to catch any detached orphaned processes..."
    # Fallback to general process killing just in case the PID file was deleted
    pkill -f "streamlit run app.py"
    echo "✅ Any lingering instances have been purged from the system."
fi
