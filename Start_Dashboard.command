#!/bin/bash
# Double-click to start the Muse EEG Dashboard.
# Sets up Python the first time, connects to the Muse and opens the dashboard window.
# Keep this Terminal window open while recording; closing it stops the dashboard
# (a running session is stopped and saved first).
cd "$(dirname "$0")"
PORT=8765
URL="http://127.0.0.1:$PORT"

open_window() {
  for app in "Google Chrome" "Microsoft Edge" "Brave Browser" "Chromium"; do
    if [ -d "/Applications/$app.app" ] || [ -d "$HOME/Applications/$app.app" ]; then
      open -na "$app" --args --app="$URL" --window-size=1480,960
      return
    fi
  done
  open "$URL"
}

echo "=== Muse EEG Dashboard ==="

if curl -s -m 1 "$URL/api/ping" >/dev/null 2>&1; then
  echo "Already running. Opening the window..."
  open_window
  exit 0
fi

if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up Python (takes a minute)..."
  PY=$(command -v python3)
  [ -z "$PY" ] && { echo "python3 not found. Install it from https://www.python.org/downloads/macos/"; read -p "Press Enter to close"; exit 1; }
  "$PY" -m venv .venv || { echo "Could not create .venv"; read -p "Press Enter to close"; exit 1; }
  .venv/bin/python -m pip install -q --upgrade pip
fi
if ! .venv/bin/python -c "import muselsl, pylsl, numpy, scipy" >/dev/null 2>&1; then
  echo "Installing packages..."
  .venv/bin/pip install -q -r requirements.txt || { echo "Package install failed."; read -p "Press Enter to close"; exit 1; }
fi

echo "Turn the Muse on and make sure it is NOT connected to the Muse phone app."
echo "If macOS asks for Bluetooth access for Terminal, click Allow."
echo ""

# open the window once the server answers
(
  for _ in $(seq 1 40); do
    if curl -s -m 1 "$URL/api/ping" >/dev/null 2>&1; then open_window; exit 0; fi
    sleep 0.5
  done
  echo "Server did not start - see the messages above."
) &

.venv/bin/python dashboard/server.py --port $PORT
echo ""
echo "Dashboard stopped. You can close this window."
