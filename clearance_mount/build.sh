#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
"${CAD_PYTHON:-.venv/bin/python}" clearance_mount/src/easy_mount_assembly.py
"${CAD_PYTHON:-.venv/bin/python}" clearance_mount/checks/build_outputs.py
"${CAD_PYTHON:-.venv/bin/python}" clearance_mount/checks/check_shell_capture.py
"${CAD_PYTHON:-.venv/bin/python}" clearance_mount/checks/check_attachment.py
