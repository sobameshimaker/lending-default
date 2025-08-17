#!/usr/bin/env bash
set -euo pipefail
python -m pip install -q -r ./requirements.txt
cp ../data/train.csv .
cp ../data/test.csv .
python ./main.py
