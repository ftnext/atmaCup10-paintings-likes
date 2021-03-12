import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import nltk
from fasttext import load_model
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from custom_texthero import (
    fillna,
    lowercase,
    remove_diacritics,
    remove_digits,
    remove_punctuation,
    remove_stopwords_func,
)


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
        if not categories:
            continue
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


def identify_language(text: str) -> Optional[str]:
    """
    >>> identify_language("Still Life")  # doctest: +SKIP
    '__label__en'
    >>> identify_language("")
    """
    if text == "":
        return None  # ""はEnglishとして判定された（誤りになりそう）
    # \nを含むと ValueError: predict processes one line at a time (remove '\n')
    pred = LANGUAGE_IDENTIFIER.predict(re.sub("\n", " ", text))
    return pred[0][0]


def preprocess_language_information(rows, fields):
    features = []
    for row in rows:
        new_row = {}
        for field in fields:
            # 空文字列（欠損値）の処理結果は、空文字列として欠損を表す
            new_row[f"{field}__lang"] = identify_language(row[field]) or ""
        features.append(new_row)
    return features


def normalize(raw_text, functions):
    for func in functions:
        raw_text = func(raw_text)
    return raw_text


def clean_text(raw_text):
    cleansing_functions = (
        fillna,
        lowercase,
        remove_diacritics,
        remove_digits,
        remove_punctuation,
        remove_stopwords_func(CUSTOM_STOPWORDS),
    )
    return normalize(raw_text, cleansing_functions)


def normalize_text_features(rows, fields):
    features = []
    for row in rows:
        new_row = {}
        for field in fields:
            text = clean_text(row[field])
            new_row[field] = text
        features.append(new_row)
    return features


def create_tfidf_features(texts_train, texts_test):
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10000)),
            ("svd", TruncatedSVD(n_components=50)),
        ]
    )
    # TODO: len(pipeline.named_steps["tfidf"].vocabulary_) -> 18390
    features_train = pipeline.fit_transform(texts_train)
    features_test = pipeline.transform(texts_test)
    return features_train, features_test


def tfidf_to_row(tfidf_array, field):
    features = []
    for row in tfidf_array:
        new_row = {}
        for idx, value in enumerate(row):
            new_row[f"{field}_tfidf_{idx}"] = value
        features.append(new_row)
    return features


def convert_text_to_features(rows_train, rows_test, field):
    texts_train = [row[field] for row in rows_train]
    texts_test = [row[field] for row in rows_test]

    features_train, features_test = create_tfidf_features(
        texts_train, texts_test
    )

    return tfidf_to_row(features_train, field), tfidf_to_row(
        features_test, field
    )


def merge(rows1, rows2):
    """
    >>> rows1 = [{"a": 100}, {"a": 50}]
    >>> rows2 = [{"b": "foo"}, {"b": "bar"}]
    >>> actual = merge(rows1, rows2)
    >>> expected = [{"a": 100, "b": "foo"}, {"a": 50, "b": "bar"}]
    >>> assert actual == expected
    >>> assert rows1 == [{"a": 100}, {"a": 50}]  # copyしたので中身は変わらない
    """
    assert len(rows1) == len(
        rows2
    ), f"Length not equal: {len(rows1)}, {len(rows2)}"
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
            ("title", "long_title", "sub_title", "more_title", "description"),
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


def add_features(rows_train, rows_test):
    added = {}
    for key, rows in {"train": rows_train, "test": rows_test}.items():
        new_features = preprocess_language_information(
            rows, ("title", "long_title", "more_title", "description")
        )
        new_rows = merge(rows, new_features)
        added[key] = new_rows
    return added["train"], added["test"]


def load_data(file_path):
    with file_path.open() as fin:
        reader = csv.DictReader(fin)
        return list(reader)


def dump_data(file_path, rows):
    field_names = rows[0].keys()
    with file_path.open("w") as fout:
        writer = csv.DictWriter(fout, field_names)
        writer.writeheader()
        writer.writerows(rows)


def preprocess_data_files(input_root, output_root, intermediate_root=None):
    rows_train = load_data(input_root / "train.csv")
    rows_test = load_data(input_root / "test.csv")

    rows_train, rows_test = add_features(rows_train, rows_test)
    if intermediate_root:
        dump_data(intermediate_root / "train.csv", rows_train)
        dump_data(intermediate_root / "test.csv", rows_test)

    min_count = 20
    features_train, features_test = create_one_hot_encoding(
        rows_train,
        rows_test,
        (
            ("acquisition_method", 20),
            ("principal_maker", 20),
            ("title__lang", 10),
            ("long_title__lang", 10),
            ("more_title__lang", 10),
            ("description__lang", 10),
            ("title", min_count),
            ("description", min_count),
            ("long_title", min_count),
            ("principal_or_first_maker", min_count),
            ("sub_title", min_count),
            ("copyright_holder", min_count),
            ("more_title", min_count),
            ("acquisition_date", min_count),
            ("acquisition_credit_line", min_count),
            ("dating_presenting_date", min_count),
            ("dating_sorting_date", min_count),
            ("dating_period", min_count),
            ("dating_year_early", min_count),
            ("dating_year_late", min_count),
        ),
    )

    text_fields = ("description",)
    normalized_train = normalize_text_features(rows_train, text_fields)
    normalized_test = normalize_text_features(rows_test, text_fields)

    rows_train = preprocess(rows_train)
    rows_train = merge(rows_train, features_train)
    rows_test = preprocess(rows_test)
    rows_test = merge(rows_test, features_test)

    for field in text_fields:
        text_features_train, text_features_test = convert_text_to_features(
            normalized_train, normalized_test, field
        )
        rows_train = merge(rows_train, text_features_train)
        rows_test = merge(rows_test, text_features_test)

    dump_data(output_root / "train.csv", rows_train)
    dump_data(output_root / "test.csv", rows_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--intermediate_root", type=Path)
    parser.add_argument(
        "--pretrained_language_identifier",
        default="models/pretrained/lid.176.bin",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.intermediate_root:
        args.intermediate_root.mkdir(parents=True, exist_ok=True)

    LANGUAGE_IDENTIFIER = load_model(args.pretrained_language_identifier)
    CUSTOM_STOPWORDS = nltk.corpus.stopwords.words(
        "dutch"
    ) + nltk.corpus.stopwords.words("english")

    preprocess_data_files(
        args.input_root, args.output_root, args.intermediate_root
    )
