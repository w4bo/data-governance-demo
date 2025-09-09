#!/bin/bash
set -exo
cd /home
for script in $(ls *.py | sort); do
    python "$script"
done
