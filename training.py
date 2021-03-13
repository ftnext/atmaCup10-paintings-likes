import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

from models.lightgbm import fit_lgbm
from models.lightgbm import params as lgbm_params


def load_target_variable(input_path):
    with input_path.open() as f:
        reader = csv.DictReader(f)
        likes = [int(row["likes"]) for row in reader]
    return np.log1p(likes)  # log(1 + x)


def load_features(input_path, float_fields):
    data = []
    with input_path.open() as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            array = []
            for key, value in row.items():
                if not value:
                    array.append(np.nan)
                elif key in float_fields:
                    array.append(float(row[key]))
                else:
                    array.append(int(row[key]))
            data.append(array)
    return np.array(data)


def get_columns(input_path):
    with input_path.open() as f:
        reader = csv.DictReader(f)
        return tuple(next(reader).keys())


def revert_to_real(y_log):
    _pred = np.expm1(y_log)
    _pred = np.where(_pred < 0, 0, _pred)  # likeは0以上
    return _pred


def visualize_oof_pred_distribution(oof, pred):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(pred, label="Test Predict", ax=ax, color="black")
    sns.histplot(oof, label="Out of Fold", ax=ax, color="C1")
    ax.legend()
    ax.grid()
    return fig, ax


def visualize_importance(models, feature_columns, top_n=50):
    feature_importance = defaultdict(float)
    for i, model in enumerate(models):
        for c, i in zip(feature_columns, model.feature_importances_):
            feature_importance[c] += i
    print(f"features count: {len(feature_importance)}")
    orders = sorted(
        feature_importance.items(), key=lambda d: d[1], reverse=True
    )
    top_ns = orders[:top_n]

    for column, importance in top_ns:
        print(f"{column}\t{importance}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_root", type=Path)
    parser.add_argument("target_containing_file", type=Path)
    parser.add_argument("submission_root", type=Path)
    parser.add_argument("--y_max", type=float)
    parser.add_argument("--check_y_max", action="store_true")
    args = parser.parse_args()

    args.submission_root.mkdir(parents=True, exist_ok=True)

    y = load_target_variable(args.target_containing_file)

    float_fields = (
        "dating_year_early",
        "dating_year_late",
        "size_h",
        "size_w",
        *[f"description_tfidf_{i}" for i in range(50)],
        *[f"material_vector_{i}" for i in range(20)],
        *[f"object_collection_vector_{i}" for i in range(3)],
        *[f"technique_vector_{i}" for i in range(8)],
        *[f"production_place_vector_{i}" for i in range(30)],
        *[f"material__object_collection_vector_{i}" for i in range(20)],
        *[f"material__technique_vector_{i}" for i in range(20)],
        *[f"material__production_place_vector_{i}" for i in range(30)],
        *[f"object_collection__technique_vector_{i}" for i in range(8)],
        *[
            f"object_collection__production_place_vector_{i}"
            for i in range(30)
        ],
        *[f"technique__production_place_vector_{i}" for i in range(30)],
        # *[
        #     f"material__object_collection__technique_vector_{i}"
        #     for i in range(20)
        # ],
        # *[
        #     f"material__object_collection__production_place_vector_{i}"
        #     for i in range(30)
        # ],
        # *[
        #     f"material__technique__production_place_vector_{i}"
        #     for i in range(30)
        # ],
        # *[
        #     f"object_collection__technique__production_place_vector_{i}"
        #     for i in range(30)
        # ],
        # *[
        #     f"material__object_collection__technique__production_place_vector_{i}"
        #     for i in range(30)
        # ],
        # *[f"description_bert_vector_{i}" for i in range(768)],
    )
    X = {}
    for data_type in ("train", "test"):
        X[data_type] = load_features(
            args.features_root / f"{data_type}.csv", float_fields
        )
    else:
        feature_columns = get_columns(args.features_root / f"{data_type}.csv")
    print(f"train: {X['train'].shape}")
    print(f"test: {X['test'].shape}")

    fold = KFold(n_splits=5, shuffle=True, random_state=71)
    cv = list(fold.split(X["train"], y))

    if args.check_y_max:
        out_of_folds = {}
        for y_max in [5.5, 6.0, 6.5, 7.0, 7.5, 8.0, None]:
            oof, _ = fit_lgbm(
                X["train"],
                y,
                cv,
                y_max=y_max,
                params=lgbm_params,
                verbose=None,
            )
            out_of_folds[y_max] = oof
        scores = []
        for key, value in out_of_folds.items():
            score_i = mean_squared_error(y, value) ** 0.5
            scores += [(key, score_i)]
        for key, score in sorted(scores, key=lambda d: d[1]):
            print(key, score)
    else:
        oof, models = fit_lgbm(
            X["train"],
            y,
            cv,
            y_max=args.y_max,
            params=lgbm_params,
            verbose=500,
        )

        pred = np.array([model.predict(X["test"]) for model in models])
        pred = np.mean(pred, axis=0)
        pred = revert_to_real(pred)

        submission_file_name = (
            f"{datetime.now():%Y%m%d-%H%M%S}_"
            f"{re.sub('/', '_', str(args.features_root))}"
        )
        submission_file = args.submission_root / f"{submission_file_name}.csv"
        with submission_file.open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["likes"])
            writer.writerows([p] for p in pred.tolist())

        fig, ax = visualize_oof_pred_distribution(oof, np.log1p(pred))
        fig.savefig(args.submission_root / f"{submission_file_name}.png")

        visualize_importance(models, feature_columns)
