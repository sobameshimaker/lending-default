\Continue = 'Stop'
# data ディレクトリ（フラット/ネスト両対応）
\ = Join-Path \ 'no1\..\data'
\ = if (Test-Path (Join-Path \ 'train.csv')) { Join-Path \ 'train.csv' } else { Join-Path \ 'train\train.csv' }
\  = if (Test-Path (Join-Path \ 'test.csv'))  { Join-Path \ 'test.csv'  } else { Join-Path \ 'test\test.csv'  }

Copy-Item \ -Destination (Join-Path \ 'no1\train.csv') -Force
Copy-Item \  -Destination (Join-Path \ 'no1\test.csv')  -Force

# Windows の Python を使って依存を入れて実行
python -m pip install -q -r (Join-Path \ 'no1\requirements.txt')
python (Join-Path \ 'no1\main.py')