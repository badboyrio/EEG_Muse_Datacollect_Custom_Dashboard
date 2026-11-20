#!/bin/bash
# One-time setup: creates a private Python environment and installs muselsl, pylsl, numpy.
cd "$(dirname "$0")"
echo "=== Muse EEG Study - setup ==="

PY=$(command -v python3)
if [ -z "$PY" ]; then
  echo "python3 not found. Install Python from https://www.python.org/downloads/macos/ and run this again."
  read -p "Press Enter to close"; exit 1
fi
echo "Using: $PY ($("$PY" --version))"

TKV=$("$PY" -c "import tkinter; print(tkinter.TkVersion)" 2>/dev/null)
if [ -z "$TKV" ]; then
  echo "tkinter is missing. Homebrew Python: run  brew install python-tk  then run this again."
  echo "Or install Python from python.org (includes tkinter)."
  read -p "Press Enter to close"; exit 1
fi
echo "tkinter OK (Tk $TKV)"
if [ "$TKV" == "8.5" ]; then
  echo "WARNING: Tk 8.5 (Apple's built-in Python) is old and can show a blank window."
  echo "         Recommended: install Python from python.org."
fi

"$PY" -m venv .venv || { echo "Could not create .venv"; read -p "Press Enter"; exit 1; }
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt || { echo "Package install failed."; read -p "Press Enter"; exit 1; }

python -c "import pylsl; print('pylsl OK, liblsl', pylsl.library_version())" || {
  echo "pylsl can't find the LSL library. Fix:  brew install labstreaminglayer/tap/lsl"
}

echo ""
echo "Setup complete. Next: double-click 1_Start_Stream.command"
read -p "Press Enter to close"
