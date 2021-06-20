"""Preprocess texts

This module preprocesses text data

Examples:
  python text.py --help
  python text.py \
    --resource <Resource file> \
    --token <Output token file>
    --doc <Output document file>

Attributes:
  deliminators: list of deliminators
  emoji_pattern: regex for emoji pattern
  filter_words: list of filter words

"""
from functools import reduce
import pickle
import re

import click
from nltk.stem import PorterStemmer


with open('data/external/delim.pickle', 'rb') as file:
    deliminators = pickle.load(file)

emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        "]+", flags=re.UNICODE)

filter_words = list(map(str,range(10))) + ['$', '%']

@click.command()
@click.option('--resource', '-r', help='Resource file', type=click.Path(), required=True)
@click.option('--token', '-t', help='Output token file', type=click.Path(), required=True)
@click.option('--doc', '-d', help='Output document file', type=click.Path(), required=True)
def preprocess(resource: str, token_file: str, doc_file: str) -> None:
    """Proprocess text data

    Args:
      resource: Resource file
      token_file: Output token file
      doc_file: Output document file

    Return:
      None

    """
    with open(resource, 'rb') as rfile:
        tokens = pickle.load(rfile)
    tokens = [[t.strip().lower() for t in d] for d in tokens] # strip tokens
    doc = [further_filter(further_split(d)) for d in tokens]

    stemmer = PorterStemmer()
    doc_stem = [[stemmer.stem(t) for t in d] for d in doc]
    token_stem = reduce(lambda x, y: x | set(y), doc_stem, set())

    with open(token_file, 'wb') as ofile:
        pickle.dump(token_stem, ofile)
    with open(doc_file, 'wb') as ofile:
        pickle.dump(doc_stem, ofile)

def further_split(doc: list) -> list:
    """Split tokens

    This function split tokens again more precisely,
    since Chinese tokenizer does not perform well with English and numbers

    Args:
      doc: list of tokens in documents

    Returns:
      List of tokens in documents after further tokenization, e.g.,
      [ 'luke', '是', '一', '位', '罹患', '精神', '疾病',
        '長期', '為', '疾患', '所', '困', '的', '大學生' ]

    """
    return reduce(lambda x, y: x+deli(remove_emoji(full2half(y))).split(), doc, [])

def full2half(string: str) -> str:
    """Convert string from full width to half width

    ref: https://segmentfault.com/a/1190000006197218

    Args:
      string: string to converted

    Returns:
      Converted string

    """
    characters = []
    for char in string:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        characters.append(num)
    return ''.join(characters)

def remove_emoji(string: str) -> str:
    """Remove emoji from string

    Args:
      string: string to converted

    Returns:
      Converted string

    """
    return emoji_pattern.sub(r'', string)

def deli(string: str) -> str:
    """Replace deliminators by space

    Args:
      string: string to converted

    Returns:
      Converted string

    """
    return reduce(lambda x, y: x.replace(y, ' '), deliminators, string)

def further_filter(doc: list) -> list:
    """Filter tokens

    Filter out the tokens that contain filter words

    Args:
      doc: list of tokens in document

    Returns:
      List of tokens without filter words

    """
    return [s for s in doc if no_filter_word(s)]

def no_filter_word(string: str) -> bool:
    """Check if the string does not contain any filter word

    Args:
      string: string to check

    Returns:
      True if the the string does not contain any filter word, and vice versa

    """
    for word in filter_words:
        if word in string:
            return False
    return True

if __name__ == '__main__':
    preprocess()
