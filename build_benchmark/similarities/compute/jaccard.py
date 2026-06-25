import re
from stop_words import STOP_WORDS

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9^_]+")


def jaccard_similarity(A: set[str], B: set[str]) -> float:
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def tokenize(text, stopwords=STOP_WORDS) -> list[str]:
    text = text.lower()
    preproc_toks = TOKEN_PATTERN.findall(text)

    tokens = [tok for tok in preproc_toks if tok not in stopwords]

    return tokens
