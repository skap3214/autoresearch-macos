#!/bin/bash
set -e

echo "=== Setting up autoresearch-macos Rubik's cube solver ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install Python dependencies
echo "Installing dependencies..."
uv sync

# Clone the dwalton solver (teacher oracle) alongside this repo
SOLVER_DIR="$(dirname "$(pwd)")/rubiks-cube-NxNxN-solver"
if [ ! -d "$SOLVER_DIR" ]; then
    echo "Cloning dwalton solver to $SOLVER_DIR..."
    git clone https://github.com/dwalton76/rubiks-cube-NxNxN-solver.git "$SOLVER_DIR"
else
    echo "Solver already present at $SOLVER_DIR"
fi

# Generate training data
echo "Generating training data..."
uv run prepare.py --force

echo ""
echo "=== Setup complete! ==="
echo "Run training:  uv run train.py"
echo "Run with log:  uv run train.py > run.log 2>&1"
