import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder


def preprocess_numeric_values(rows, fields_types_map):
    features = []
    for row in rows:
        new_row = {}
        for field, _type in fields_types_map.items():
            value_str = row[field]
            if not value_str:
                new_row[field] = ""
            else:
                new_row[field] = _type(row[field])
        features.append(new_row)
    return features


def preprocess_text_values(rows, fields):
    """
    >>> rows = [
    ...     {"title": "Awesome title", "long_title": "Long text", "dating_period": 19},
    ...     {"title": "Title", "long_title": "Looong text", "dating_period": 17}
    ... ]
    >>> actual = preprocess_text_values(rows, ("title",))
    >>> expected = [
    ...     {"StringLength__title": 13}, {"StringLength__title": 5}
    ... ]
    >>> assert actual == expected
    """
    features = []
    for row in rows:
        new_row = {}
        for field in fields:
            new_row[f"StringLength__{field}"] = len(row[field])
        features.append(new_row)
    return features


def create_count_encoding_feature(rows, fields):
    counter_map = {}
    for field in fields:
        counter_map[field] = Counter(row[field] for row in rows)

    features = []
    for row in rows:
        new_row = {}
        for field in fields:
            value = row[field]
            new_row[f"CE__{field}"] = counter_map[field][value]
        features.append(new_row)
    return features


def create_one_hot_encoding(rows_train, rows_test, fields_thresholds):
    features_train = [{} for _ in range(len(rows_train))]
    features_test = [{} for _ in range(len(rows_test))]

    for field, threshold in fields_thresholds:
        new_rows_train_per_field = []
        new_rows_test_per_field = []

        field_values = [row[field] for row in rows_train]
        counter = Counter(field_values)
        categories = [
            item for item, count in counter.most_common() if count > threshold
        ]
        encoder = OneHotEncoder(
            categories=[categories], handle_unknown="ignore"
        )
        transformed_train = encoder.fit_transform([[v] for v in field_values])
        transformed_test = encoder.transform(
            [[row[field]] for row in rows_test]
        )

        # TODO: catagories_はlistなので複数同時に与えられるのかもしれない
        encoder_categories = tuple(encoder.categories_[0])
        for row in transformed_train.toarray():
            new_row = {}
            for c, v in zip(encoder_categories, row):
                new_row[f"{field}={c}"] = int(v)
            new_rows_train_per_field.append(new_row)
        features_train = merge(features_train, new_rows_train_per_field)

        for row in transformed_test.toarray():
            new_row = {}
            for c, v in zip(encoder_categories, row):
                new_row[f"{field}={c}"] = int(v)
            new_rows_test_per_field.append(new_row)
        features_test = merge(features_test, new_rows_test_per_field)

    return features_train, features_test


def length_mm(length_str, length_unit):
    if length_unit == "cm":
        return float(length_str) * 10
    else:  # case "mm". consider "" to "mm".
        return float(length_str)


def search_length_pattern(string, prefix):
    number_pattern = r"(\d+|\d+\.\d+)"
    unit_pattern = r"(cm|mm)"
    first_pattern = f"{prefix} {number_pattern}{unit_pattern}"
    second_pattern = f"{prefix} {number_pattern}"  # missing unit case

    if m := re.search(first_pattern, string):
        return m[1], m[2]
    elif m := re.search(second_pattern, string):
        return m[1], None
    return None, None


def extract_sizes_mm(size_str):
    """
    >>> extract_sizes_mm("h 105mm × w 63mm")
    (105.0, 63.0)
    >>> extract_sizes_mm("h mm × w mm")
    (None, None)
    >>> extract_sizes_mm("h 11cm × w 9.5cm")
    (110.0, 95.0)
    >>> extract_sizes_mm("w 180mm × h 266mm")
    (266.0, 180.0)
    >>> extract_sizes_mm("h 196.8cm × w 74.3cm × d 4.5cm")
    (1968.0, 743.0)
    >>> extract_sizes_mm("h 52.1cm")
    (521.0, None)
    >>> extract_sizes_mm("h 105mm × w mm")
    (105.0, None)
    >>> extract_sizes_mm("h 47mm × w 81")
    (47.0, 81.0)
    >>> extract_sizes_mm("h 116")
    (116.0, None)
    """
    height, width = None, None

    height_str, height_unit = search_length_pattern(size_str, "h")
    if height_str:
        height = length_mm(height_str, height_unit)

    width_str, width_unit = search_length_pattern(size_str, "w")
    if width_str:
        width = length_mm(width_str, width_unit)

    return height, width


def preprocess_subtitle(rows):
    features = []
    for row in rows:
        height, width = extract_sizes_mm(row["sub_title"])
        new_row = {"size_h": height or "", "size_w": width or ""}
        features.append(new_row)
    return features


def merge(rows1, rows2):
    """
    >>> rows1 = [{"a": 100}, {"a": 50}]
    >>> rows2 = [{"b": "foo"}, {"b": "bar"}]
    >>> actual = merge(rows1, rows2)
    >>> expected = [{"a": 100, "b": "foo"}, {"a": 50, "b": "bar"}]
    >>> assert actual == expected
    >>> assert rows1 == [{"a": 100}, {"a": 50}]  # copyしたので中身は変わらない
    """
    merged = []
    for row1, row2 in zip(rows1, rows2):
        row = row1.copy()
        row.update(row2)
        merged.append(row)
    return merged


def preprocess(rows):
    preprocess_functions = [
        (
            preprocess_numeric_values,
            {
                "dating_period": int,
                "dating_year_early": float,
                "dating_year_late": float,
            },
        ),
        (preprocess_subtitle, ()),
        (
            preprocess_text_values,
            ("title", "long_title", "more_title", "description"),
        ),
        (
            create_count_encoding_feature,
            ("acquisition_method", "title", "principal_maker"),
        ),
    ]

    output_rows = [{} for _ in range(len(rows))]
    for func, fields in preprocess_functions:
        if fields:
            preprocessed = func(rows, fields)
        else:
            preprocessed = func(rows)
        assert len(preprocessed) == len(rows), func.__name__

        output_rows = merge(output_rows, preprocessed)

    return output_rows


def preprocess_data_files(input_root, output_root):
    with (input_root / "train.csv").open() as f_train, (
        input_root / "test.csv"
    ).open() as f_test:
        reader_train = csv.DictReader(f_train)
        rows_train = list(reader_train)
        reader_test = csv.DictReader(f_test)
        rows_test = list(reader_test)

    features_train, features_test = create_one_hot_encoding(
        rows_train,
        rows_test,
        (("acquisition_method", 20), ("principal_maker", 20)),
    )

    rows_train = preprocess(rows_train)
    rows_train = merge(rows_train, features_train)
    rows_test = preprocess(rows_test)
    rows_test = merge(rows_test, features_train)

    field_names = rows_train[0].keys()
    with (output_root / "train.csv").open("w") as f_train, (
        output_root / "test.csv"
    ).open("w") as f_test:
        writer_train = csv.DictWriter(f_train, field_names)
        writer_train.writeheader()
        writer_train.writerows(rows_train)
        writer_test = csv.DictWriter(f_test, field_names)
        writer_test.writeheader()
        writer_test.writerows(rows_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    preprocess_data_files(args.input_root, args.output_root)
