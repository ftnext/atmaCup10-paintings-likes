"""
# #は先頭から除き、%は末尾から除く（bashの変数参照）
for data_csv_path in data/datasets/*.csv
do
  data_csv_file=${data_csv_path#data/datasets/}
  python eda/facet.py ${data_csv_path} data/facets/${data_csv_file%.csv}
done
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_csv")
    parser.add_argument("output_root_dir")
    args = parser.parse_args()

    output_root_path = Path(args.output_root_dir)
    output_root_path.mkdir(parents=True, exist_ok=True)

    # head -n 2 data/datasets/*.csv で全てヘッダーがあることを確認
    with open(args.data_csv) as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)

    columns = rows[0].keys()
    for column in columns:
        # csvなのでrow[column]は文字列。欠損の場合は空文字''になる
        # csvの中にNaNという文字列は登場しないと仮定して、欠損していることが分かるように置き換える
        values = [row[column] if row[column] else "NaN" for row in rows]
        counter = Counter(values)
        with open(output_root_path / f"{column}.csv", "w") as fout:
            writer = csv.writer(fout)
            writer.writerows(counter.most_common())
