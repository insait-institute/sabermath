import re

import pya0

_MATH_BLOCK_RE = re.compile(
    r'\$\$(.*?)\$\$'     # $$...$$
    r'|\$(.*?)\$'        # $...$
    r'|\\\((.*?)\\\)'    # \(...\)
    r'|\\\[(.*?)\\\]',   # \[...\]
    re.DOTALL,
)
_WORD_RE = re.compile(r"\b[a-zA-Z]+\b")

def math_word_tokens(text: str, lowercase: bool = True) -> list[str]:
    # Math: tokenize each LaTeX block with pya0
    # pya0.tokenize() returns plain str tokens for most nodes, but structured
    # nodes (e.g. quantified/wildcard "qvar" variables) come back as tuples
    # like (token, "qvar"). Flatten those to strings so every token is a str -
    # sklearn's vocabulary sorting can't compare str and tuple.
    math_blocks = [
        next(g for g in m.groups() if g is not None)
        for m in _MATH_BLOCK_RE.finditer(text)
    ]
    math_tokens = [
        tok if isinstance(tok, str) else str(tok[0])
        for block in math_blocks
        for tok in pya0.tokenize(block)
    ]

    # Words: strip math first so LaTeX letters don't leak into prose tokens
    words_string = _MATH_BLOCK_RE.sub(" ", text)
    if lowercase:
        words_string = words_string.lower()
    words = _WORD_RE.findall(words_string)

    return math_tokens + words