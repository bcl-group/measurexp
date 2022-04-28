import pandas as pd
import re
import argparse
import os
import sys

def _col2muscles_name(col_muscle: str):
    match = re.match(r'^(.+):', col_muscle)
    if match is not None:
        muscles_name = match.group(1)
    return muscles_name

def to_pretty(df: pd.DataFrame):
    # カラム名取得
    col_muscles = df.columns.tolist()[1:]
    # 新しいカラム名を定義
    df.columns = ['Time [s]'] \
        + [_col2muscles_name(_) for _ in col_muscles]
    # df.loc[:, 'タスク名'] = self.taskname
    # インデックスの設定
    df.set_index('Time [s]', inplace=True)

if __name__ == '__main__':
    # filename = '/mnt/d/experiment/results/202103/A/EMG-0111.csv'
    parser = argparse.ArgumentParser(description='Convert raw EMG data together.')
    parser.add_argument("file", help='CSV file to be converted.')
    parser.add_argument("--header", help='Row number to use as the column names, and the start of the data. (default=116)')
    parser.add_argument("--encoding", help='File character code. (default="Shift-JIS")')
    parser.add_argument("-o", "--output", help="Place the output into csv file.")
    args = parser.parse_args()
    
    header = 116 if args.header == None else int(args.header)
    encoding = 'Shift-JIS' if args.encoding == None else args.encoding

    if not os.path.isfile(args.file):
        print(f'{args.file} が存在しません', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.file, encoding=encoding, header=header)
    to_pretty(df)
    
    if args.output == None:
        print(df)
    else:
        df.to_csv(args.output)


