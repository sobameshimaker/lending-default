#!/usr/bin/env bash
set -euo pipefail
python -m pip install -q -r ./requirements.txt
DATA_DIR="\"
TRAIN="\/train.csv"
TEST="\/test.csv"
[[ -f "\" ]] || TRAIN="\/train/train.csv"
[[ -f "\"  ]] || TEST="\/test/test.csv"
cp "\" ./train.csv
cp "\"  ./test.csv
python ./main.py