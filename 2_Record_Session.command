#!/bin/bash
# Runs one recording session (about 22 minutes). Needs 1_Start_Stream running.
cd "$(dirname "$0")"
source .venv/bin/activate || { echo "Run 0_Setup.command first."; read -p "Press Enter"; exit 1; }

echo "=== Record session ==="
ls data 2>/dev/null | grep sub- >/dev/null && { echo "Existing recordings:"; ls -d data/sub-*/ses-* 2>/dev/null; echo ""; }

read -p "Subject ID (e.g. 01): " SUB
read -p "Session ID (e.g. 01): " SES
if [ -z "$SUB" ] || [ -z "$SES" ]; then echo "Both IDs are required."; read -p "Press Enter"; exit 1; fi

python record_session.py --sub "$SUB" --ses "$SES"

OUT="data/sub-$SUB/ses-$SES"
[ -d "$OUT" ] && open "$OUT"
read -p "Press Enter to close"
