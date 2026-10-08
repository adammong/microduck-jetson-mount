#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export CADGEN_CACHE_DIR="$PWD/tmp/cad-cache"
export CADGEN_DAEMON_STATE_DIR="$PWD/tmp/cad-daemon"
export CADGEN_DAEMON=0
export XDG_CACHE_HOME="$PWD/tmp/cache"
"${CAD_PYTHON:-.venv/bin/python}" easy_mount/src/easy_mount_assembly.py
"${CAD_PYTHON:-.venv/bin/python}" easy_mount/checks/build_outputs.py
