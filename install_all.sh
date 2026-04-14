#!/usr/bin/env bash
# Install all oceanbench sub-packages in editable mode.
# Usage: source oceanbench-venv/bin/activate && bash install_all.sh

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

pip install -e "$DIR/oceanbench-core"
pip install -e "$DIR/oceanbench-env"
pip install -e "$DIR/oceanbench-models"
pip install -e "$DIR/oceanbench-tasks"
pip install -e "$DIR/oceanbench-bench"
pip install -e "$DIR/oceanbench-policies"
pip install -e "$DIR/oceanbench-viz"
pip install -e "$DIR/oceanbench-agents"
pip install -e "$DIR/oceanbench-data-provider"

echo "All oceanbench packages installed successfully."
