#!/bin/bash
set -exo

echo "Running all Python scripts in alphabetical order..."

for script in $(ls *.py | sort); do
    echo ">>> Running $script"
    python "$script"
done

echo "All scripts completed."