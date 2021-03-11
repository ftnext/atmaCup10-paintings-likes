"""Re-implement texthero.preprocessing for not pandas raw texts.

ref: https://github.com/jbesomi/texthero/blob/5a83c6ebebee46a7004b7ba141bc6b9f6d744790/texthero/preprocessing.py
"""

from __future__ import annotations

import re
import string
import unicodedata
from functools import partial
from typing import Optional

DIGITS_BLOCK_PATTERN = re.compile(r"\b\d+\b")  # word made by only digits
DIGITS_PATTERN = re.compile(r"\d+")

# string.punctuation の文字の1つ以上の繰返し（キャプチャの()は不要）
PUNCTUATION_PATTERN = re.compile(rf"[{string.punctuation}]+")

WORD_PATTERN = re.compile(
    r"""(?x)                              # Set flag to allow verbose regexps
  \w+(?:-\w+)*                           # Words with optional internal hyphens
  | \s*                                   # Any space
  | [][!"#$%&'*+,-./:;<=>?@\\^():_`{|}~]  # Any symbol
"""
)


def fillna(raw_text: Optional[str]) -> str:
    """
    >>> fillna("I'm")
    "I'm"
    >>> fillna("")
    ''
    >>> fillna(None)
    ''
    """
    if not raw_text:
        return ""
    return raw_text


def lowercase(raw_text: str) -> str:
    """
    >>> lowercase("This is NeW YoRk wIth upPer letters")
    'this is new york with upper letters'
    """
    return raw_text.lower()


def remove_digits(raw_text: str, only_blocks: bool = True) -> str:
    """
    >>> raw_text = "7ex7hero is fun 1111"
    >>> remove_digits(raw_text)  # change digits block to whitespace
    '7ex7hero is fun  '
    >>> remove_digits(raw_text, only_blocks=False)
    ' ex hero is fun  '
    """
    replaced_symbol = " "
    if only_blocks:
        return DIGITS_BLOCK_PATTERN.sub(replaced_symbol, raw_text)
    return DIGITS_PATTERN.sub(replaced_symbol, raw_text)


def remove_punctuation(raw_text: str) -> str:
    """
    >>> remove_punctuation("Finnaly.")
    'Finnaly '
    """
    return PUNCTUATION_PATTERN.sub(" ", raw_text)


def remove_diacritics(raw_text: str) -> str:
    """
    >>> raw_text = "Montréal, über, 12.89, Mère, Françoise, noël, 889, اِس, اُس"
    >>> remove_diacritics(raw_text)
    'Montreal, uber, 12.89, Mere, Francoise, noel, 889, اس, اس'
    """
    nfkd_form = unicodedata.normalize("NFKD", raw_text)
    return "".join(
        char for char in nfkd_form if not unicodedata.combining(char)
    )  # 結合クラスがあるcharは除かれる


def remove_stopwords(raw_text: str, stopwords: set[str]):
    """
    >>> remove_stopwords("the book of the jungle", set(["the", "of"]))
    '  book     jungle'
    """
    replaced_symbol = " "
    return "".join(
        t if t not in stopwords else replaced_symbol
        for t in WORD_PATTERN.findall(raw_text)
    )


def remove_stopwords_func(stopwords):
    """
    >>> func = remove_stopwords_func(set(["the", "of"]))
    >>> func("the book of the jungle")
    '  book     jungle'
    """
    return partial(remove_stopwords, stopwords=stopwords)
