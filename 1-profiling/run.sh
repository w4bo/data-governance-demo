#!/bin/bash
set -exo
cd /home/1-profiling
rm -rf data/bronze || true
rm -rf data/silver || true
rm -rf data/gold || true
for script in $(ls *.py | sort); do
    python "$script"
done
