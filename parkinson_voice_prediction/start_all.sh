#!/bin/bash

# Navigate to the correct directory just in case
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================="
echo " Starting Parkinson's Speech & Image App  "
echo "=========================================="

# Check if app is already running
if [ -f app_pid.txt ]; then
    if ps -p $(cat app_pid.txt) > /dev/null; then
        echo "⚠️ The application is already running! (PID: $(cat app_pid.txt))"
        echo "If you want to restart it, run ./stop_all.sh first."
        exit 1
    else
        # Stale PID file
        rm app_pid.txt
    fi
fi

# Run the Streamlit app in the background and pipe output to app_log.txt
echo "Booting up the dashboard..."
nohup python3 -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false > app_log.txt 2>&1 &

# Save the Process ID (PID)
PID=$!
echo $PID > app_pid.txt

echo "✅ App successfully deployed in the background!"
echo "📍 Dashboard is usually available at: http://localhost:8501"
echo "📝 You can monitor the live logs gracefully by running: tail -f app_log.txt"
