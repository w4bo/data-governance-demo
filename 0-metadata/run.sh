#!/bin/bash
set -exo
cd /home/0-metadata
rm -rf out || true
python detect.py --input-dir imgs/ --output-dir ./out
python upload.py