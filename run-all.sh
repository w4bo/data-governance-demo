#!/usr/bin/env bash
set -e  # stop on error
# ./init.sh
for dir in */; do
  if [ -f "$dir/run.sh" ]; then
    echo "Executing $dir/run.sh..."
    (
      cd "$dir"
      chmod +x run.sh
      ./run.sh
    )
  else
    echo "Skipping $dir (no run.sh found)"
  fi
done
