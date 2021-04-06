import re
from dataclasses import dataclass
from typing import Optional

import fasttext


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
