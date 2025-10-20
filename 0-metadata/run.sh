#!/bin/bash
set -exo
cd /home/0-metadata
rm -rf out || true
python3.13 detect.py --input-dir imgs/ --output-dir ./out
python3.13 upload.py