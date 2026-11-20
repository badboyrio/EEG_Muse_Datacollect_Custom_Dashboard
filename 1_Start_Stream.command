#!/bin/bash
# Connects to the Muse over Bluetooth and streams EEG over LSL.
# Keep this window OPEN for the whole recording.
cd "$(dirname "$0")"
source .venv/bin/activate || { echo "Run 0_Setup.command first."; read -p "Press Enter"; exit 1; }

echo "=== Muse stream ==="
echo "1. Turn the Muse on (lights blinking)."
echo "2. Make sure it is NOT connected to the Muse phone app."
echo "3. If macOS asks for Bluetooth permission for Terminal, click Allow."
echo ""
echo "When you see 'Streaming EEG...', leave this window open and"
echo "double-click 2_Record_Session.command"
echo ""

muselsl stream

echo ""
echo "Stream stopped."
read -p "Press Enter to close"
