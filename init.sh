#!/bin/bash
export TERM="xterm-256color"

# Resolve the actual directory of this script, even if it's a symlink
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

python "$SCRIPT_DIR/main.py"
