#!/usr/bin/env bash
# macOS 27 only. Starts fm serve on :1976 and fm_tool_proxy on :1977, then prints the Goose env to use.
# Runs in the foreground until either process exits (the setup script runs it from a per-user
# LaunchAgent, io.github.bobstrogg.fm-serve-goose, so it also starts at login). Logs: runs/.
# PYTHON=/path/to/python3 picks the interpreter (default: python3 on PATH).
set -e
cd "$(dirname "$0")"; mkdir -p runs
/usr/bin/fm license --status >/dev/null 2>&1 || { echo "Run: sudo fm license   (accept the FM1 notice first)"; exit 69; }
/usr/bin/fm serve --port 1976 & FM=$!
trap 'kill $FM $PX 2>/dev/null' EXIT
trap 'exit 143' TERM INT
"${PYTHON:-python3}" fm_tool_proxy.py & PX=$!
sleep 2; curl -s http://127.0.0.1:1977/proxy/version; echo
cat <<ENV
export GOOSE_PROVIDER=openai GOOSE_MODEL=system
export OPENAI_HOST=http://127.0.0.1:1977 OPENAI_BASE_PATH=v1/chat/completions
export GOOSE_CONTEXT_LIMIT=4096   # set to what 'fm' reports for your Mac (4096 or 8192)
goose run --no-profile --with-builtin developer -t "Run a shell command that prints hello"
ENV
# Stop when either child dies (bash 3.2 has no wait -n), so launchd/the caller sees it.
while kill -0 $FM 2>/dev/null && kill -0 $PX 2>/dev/null; do sleep 5 & wait $! 2>/dev/null; done
exit 1
