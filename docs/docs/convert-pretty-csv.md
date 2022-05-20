---
title: 筋電データの整形
---

筋電位データの整形は以下のように行います。

```bash
poetry shell

# CSV ファイルのある場所 (例)
dir=/mnt/d/experiment/results/202103/X

mkdir -p $dir/raw
files=$(ls $dir/*.csv)
mv $files $dir/raw

pid=""
for csv in $files; do
    bname=$(basename $csv)
    dname=$(dirname $csv)
    poetry run python -m measurexp.pretty-csv -o $csv $dname/raw/$bname &
    pid="$pid $!"
done
wait $pid

```
