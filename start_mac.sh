#!/usr/bin/env bash
# macOS 27 only (NOT tested; this repo was built on Linux against a mock).
# Starts fm serve on :1976 and fm_tool_proxy on :1977, then prints the Goose env to use.
set -e
cd "$(dirname "$0")"
fm license --status >/dev/null 2>&1 || { echo "Run: sudo fm license   (accept the FM1 notice first)"; exit 69; }
/usr/bin/fm serve --port 1976 & FM=$!
trap 'kill $FM $PX 2>/dev/null' EXIT
python3 fm_tool_proxy.py & PX=$!
sleep 2; curl -s http://127.0.0.1:1977/v1/models; echo
cat <<ENV
export GOOSE_PROVIDER=openai GOOSE_MODEL=system
export OPENAI_HOST=http://127.0.0.1:1977 OPENAI_BASE_PATH=v1/chat/completions
export GOOSE_CONTEXT_LIMIT=4096   # set to what 'fm' reports for your Mac (4096 or 8192)
goose run --no-profile --with-builtin developer -t "Run a shell command that prints hello"
ENV
wait
