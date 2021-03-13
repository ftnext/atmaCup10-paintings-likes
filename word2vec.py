"""object_id, nameの形式のCSVについて、object_idごとに固定長のベクトル表現を作る"""

import argparse
import hashlib
import os
import pickle
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from gensim.models import word2vec
from tqdm import tqdm

from preprocess import load_data

os.environ["PYTHONHASHSEED"] = "0"
N_ITER = 100


def hashfxn(x):
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16)


def dump_pickle(file_path, data):
    with file_path.open("wb") as f:
        pickle.dump(data, f)


def build_vector_map(input_root, output_root, combination_info):
    size = max(c[1] for c in combination_info)
    for data_name, _ in combination_info:
        rows = load_data(input_root / f"{data_name}.csv")
        id_to_names = defaultdict(list)
        for row in rows:
            if data_name == "production_place":
                normalized_name = re.sub(r"^\? ", "", row["name"])
                if normalized_name not in id_to_names[row["object_id"]]:
                    # 先頭の?を外したあとのnameがすでに含まれていたら重複させない
                    id_to_names[row["object_id"]].append(normalized_name)
            else:
                # ex. "001020bd00b149970f78" -> ["oil paint (paint)", "panel"]
                id_to_names[row["object_id"]].append(row["name"])

    w2v_model = word2vec.Word2Vec(
        id_to_names.values(),
        size=size,
        min_count=1,
        window=1,
        iter=N_ITER,
        workers=1,
        seed=42,
        hashfxn=hashfxn,
    )

    id_to_vector = {}
    for object_id, names in id_to_names.items():
        vectors = [w2v_model.wv[name] for name in names]
        id_to_vector[object_id] = np.mean(vectors, axis=0)

    data_combination = "__".join(c[0] for c in combination_info)
    print(data_combination, vectors[0].shape)
    output_path = output_root / f"{data_combination}.pkl"
    dump_pickle(output_path, id_to_vector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    names_data = (
        ("material", 20),
        ("object_collection", 3),
        ("technique", 8),
        ("production_place", 30),
    )
    names_combinations = []
    for i in range(1, 2 + 1):  # 2つの組合せに留める
        names_combinations.extend(combinations(names_data, i))

    for combination_info in tqdm(names_combinations):
        build_vector_map(args.input_root, args.output_root, combination_info)
