import re
import pya0


def get_math_words_tokens(math_string, words_string):

    math_blocks = re.findall(r"\${1,2}(.*?)\${1,2}", math_string, re.DOTALL)
    math_tokens = [token for block in math_blocks for token in pya0.tokenize(block)]

    words_tokens = re.findall(r"\b[a-zA-Z]+\b", words_string.lower())

    return math_tokens, words_tokens
