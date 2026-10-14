#!/bin/bash
# Double-click in Finder to start the NV-Reason-CT web UI on a Mac.
# Opens http://127.0.0.1:7860 in your default browser once the server is up.
# Keep this window open while you use the page; close it (or press Ctrl-C) to stop.
#
# Uses $NVRCT_PYTHON if set, otherwise the Python of a conda env named "nvreasonct".
# Extra arguments are passed to server.py (e.g. --mock).

cd "$(dirname "$0")" || exit 1
PORT="${NVRCT_PORT:-7860}"
URL="http://127.0.0.1:$PORT"

PY="$NVRCT_PYTHON"
if [ -z "$PY" ]; then
  bases=("$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" "/opt/homebrew/Caskroom/miniconda/base")
  if command -v conda > /dev/null; then bases=("$(conda info --base)" "${bases[@]}"); fi
  for base in "${bases[@]}"; do
    if [ -x "$base/envs/nvreasonct/bin/python" ]; then PY="$base/envs/nvreasonct/bin/python"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "Could not find the 'nvreasonct' Python environment."
  echo "Create it as described in webapp/README.md, or set NVRCT_PYTHON to a Python with the requirements installed."
  read -r -p "Press Return to close. " _
  exit 1
fi

if curl -s --max-time 2 "$URL/api/status" > /dev/null; then
  echo "NV-Reason-CT is already running; opening $URL"
  open "$URL"
  exit 0
fi

echo "Starting NV-Reason-CT. The model takes about a minute to load; the page shows when it is ready."
echo "Keep this window open while you use the page. Close it to stop the server."
"$PY" server.py --port "$PORT" "$@" &
SERVER=$!
trap 'kill $SERVER 2> /dev/null' EXIT INT TERM HUP

for _ in $(seq 1 60); do
  if curl -s --max-time 2 "$URL/api/status" > /dev/null; then
    open "$URL"
    break
  fi
  kill -0 $SERVER 2> /dev/null || break
  sleep 1
done
wait $SERVER
