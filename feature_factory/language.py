from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fasttext

# Enable to import data_io as both script and module.
if __name__ == "__main__" and __package__ is None:
    import data_io
else:
    from . import data_io


@dataclass
class LanguageIdentifier:
    _model: fasttext.FastText._FastText

    @classmethod
    def create_from(cls, model_path: str) -> "LanguageIdentifier":
        model = fasttext.load_model(model_path)
        return cls(model)

    def identify(self, text: str) -> Optional[str]:
        """
        >>> identifier = LanguageIdentifier(fasttext.FastText._FastText())
        >>> identifier.identify("Still Life")  # doctest: +SKIP
        '__label__en'
        >>> identifier.identify("")
        """
        if text == "":
            return None  # ""はEnglishとして判定された（誤りになりそう）
        # \nを含むと ValueError: predict processes one line at a time (remove '\n')
        pred = self._model.predict(re.sub("\n", " ", text))
        return pred[0][0]

    def preprocess_language_information(
        self, rows: Iterable[dict[str, str]], fields: Iterable[str]
    ) -> list[dict[str, str]]:
        """rowsのうちfieldsに含まれるキーについて、言語を判定した結果を新たに返す

        返り値の要素の順番は、rowsの要素の順番と対応。
        返り値の各要素のキーは、fieldsで指定された名前と対応する（foo -> foo__lang）
        """
        features = []
        for row in rows:
            new_row = {}
            for field in fields:
                # 空文字列（欠損値）の処理結果は、空文字列として欠損を表す
                new_row[f"{field}__lang"] = self.identify(row[field]) or ""
            features.append(new_row)
        return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--pretrained_language_identifier",
        default="models/pretrained/lid.176.bin",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    identifier = LanguageIdentifier.create_from(
        args.pretrained_language_identifier
    )

    fields = ("title", "long_title", "more_title", "description")
    for file_name in ("train.csv", "test.csv"):
        rows = data_io.load_data(args.input_root / file_name)
        new_features = identifier.preprocess_language_information(rows, fields)
        data_io.dump_data(args.output_root / file_name, new_features)
