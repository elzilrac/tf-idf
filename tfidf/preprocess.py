#!/usr/bin/env python3

"""Pre processing step for text.

Example:
    pp = Preprocesses()
"""

import re
from collections import namedtuple
from functools import lru_cache

from html import unescape

from .keyword import Keyword
from .porter2 import stem


# from cachetools import LRUCache  # python2


def handle_unicode(text):
    """Needed for the description fields."""
    if re.search(r'\\+((u([0-9]|[a-z]|[A-Z]){4}))', text):
        text = text.encode('utf-8').decode('unicode-escape')
    text = re.sub(r'\\n', '\n', text)
    text = re.sub(r'\\t', '\t', text)
    return text


def handle_html_unquote(text):
    """Detect if there are HTML encoded characters, then decode them."""
    if re.search(r'(&#?x?)([A-Z]|[a-z]|[0-9]){2,10};', text):
        text = unescape(text)
    return text


def handle_mac_quotes(text):
    """Handle the unfortunate non-ascii quotes OSX inserts."""
    text = text.replace('“', '"').replace('”', '"')\
        .replace('‘', "'").replace('’', "'")
    return text


def handle_text_break_dash(text):
    """Convert text break dashes into semicolons to simplify things.

    Example:
        "She loved icecream- mint chip especially"
        "She loved icecream - mint chip especially"
        both convert to
        "She loved icecream; mint chip especially"

        However,
        "The 27-year-old could eat icecream any day"
        will not be changed.
    """
    return re.sub(r'\s+-\s*|\s*-\s+', ';', text)


def clean_text(raw_text):
    """Strip text of non useful characters."""
    # Must strip HTML tags out first!
    text = re.sub('<[^<]+?>', '', raw_text)
    text = handle_unicode(text)
    text = handle_html_unquote(text)
    text = handle_mac_quotes(text)
    text = handle_text_break_dash(text)
    text = text.lower()

    regex_subs = ['\t\n\r', '\s+', '&']
    for regex_sub in regex_subs:
        text = re.sub(regex_sub, ' ', text)
    return text


class Preprocessor(object):
    """Prep the text for TF-IDF calculations.

    Fixes some unicode problems, handles HTML character encoding,
    and removes HTML tags.

    Strips some non alphanumeric characters, but keeps ngram boundary
    markers (eg, period (',') and semi-colon (';'))

    If a stopwords file is provided, it will remove stopwords.

    Example:
        >>> processor = Preprocessor('english_stopwords.txt')
        >>> processor.clean('He was an interesting fellow.')
        "was interesting fellow."
    """

    stopwords = set()
    contractions = r"(n't|'s|'re)$"
    negative_gram_breaks = r'[^:;!^,\?\.\[|\]\(|\)"`]+'

    def __init__(self, stopwords_file=None, stemmer=stem, gramsize=1, all_ngrams=True):
        """Preprocessor must be initalized for use if using stopwords.

        stopwords_file (filename): contains stopwords, one per line
        stemmer (function):  takes in a word and returns the stemmed version
        gramsize (int): maximum word size for ngrams
        all_ngrams (bool):
            if true, all possible ngrams of length "gramsize" and smaller will
            be examined. If false, only ngrams of _exactly_ length "gramsize"
            will be run.
        negative_gram_breaks (regex):
            if a word ends with one of these characters, an
            ngram may not cross that. Expressed as a _negative_ regex.
            Example:
            in the sentence "Although he saw the car, he ran across the street"
            "car he" may not be a bi-gram
        """
        if stopwords_file:
            self._load_stopwords(stopwords_file)
        self.__stemmer = stemmer
        self.__gramsize = gramsize
        self.__all_ngrams = all_ngrams

    @property
    def gramsize(self):
        """Number of words in the ngram."""
        return self.__gramsize

    @property
    def all_ngrams(self):
        """True if ngrams of size "gramsize" or smaller will be generated.

        False if only ngrams of _exactly_ size "gramsize" are generated.
        """
        return self.__all_ngrams

    def _load_stopwords(self, filename):
        with open(filename) as f:
            words = []
            for line in f:
                words.append(line.strip())
        self.stopwords = set(words)

    def handle_stopwords(self, text):
        """Remove stop words from the text."""
        out = []
        for word in text.split(' '):
            # Remove common contractions for stopwords when checking list
            check_me = re.sub(self.contractions, '', word)
            if check_me in self.stopwords:
                continue
            out.append(word)
        return ' '.join(out)

    def normalize_term(self, text):
        """Clean first cleans the text characters, then removes the stopwords.

        Assumes the input is already the number of words you want for the ngram.
        """
        text = clean_text(text)
        text = self.handle_stopwords(text)
        return self.stem_term(text)

    @lru_cache(maxsize=10000)
    def _stem(self, word):
        """The stem cache is used to cache up to 10,000 stemmed words.

        This substantially speeds up the word stemming on larger documents.
        """
        return self.__stemmer(word)

    def stem_term(self, term):
        """Apply the standard word procesing (eg stemming). Returns a stemmed ngram."""
        return ' '.join([self._stem(x) for x in term.split(' ')])

    def yield_keywords(self, raw_text, document=None):
        """Yield keyword objects as mono, di, tri... *-grams.

        Use this as an iterator.

        Will not create ngrams across logical sentence breaks.
        Example:
            s = "Although he saw the car, he ran across the street"
            the valid bigrams for the sentences are:
            ['Although he', 'saw the', 'he saw', 'the car',
            'he ran', 'across the', 'ran across', 'the street']
            "car he" is not a valid bi-gram

        This will also stem words when applicable.
        Example:
            s = "All the cars were honking their horns."
            ['all', 'the', 'car', 'were', 'honk', 'their', 'horn']
        """
        gramlist = range(1, self.gramsize + 1) if self.all_ngrams else [self.gramsize]

        for sentence in positional_splitter(self.negative_gram_breaks, raw_text):
            words = [x for x in positional_splitter(r'\S+', sentence.text)]
            # Remove all stopwords
            words_no_stopwords = []
            for w in words:
                # Remove common contractions for stopwords when checking list
                check_me = re.sub(self.contractions, '', w.text)
                if check_me not in self.stopwords:
                    words_no_stopwords.append(w)

            # Make the ngrams
            for gramsize in gramlist:
                # You need to try as many offsets as chunk size
                for offset in range(0, gramsize):  # number of words offest
                    data = words_no_stopwords[offset:]
                    text_in_chunks = [data[pos:pos + gramsize]
                                      for pos in range(0, len(data), gramsize)
                                      if len(data[pos:pos + gramsize]) == gramsize]
                    for word_list in text_in_chunks:
                        word_text = ' '.join([self.stem_term(w.text) for w in word_list])
                        word_global_start = sentence.start + word_list[0].start
                        word_global_end = sentence.start + word_list[-1].end
                        yield Keyword(word_text, document=document,
                                      start=word_global_start, end=word_global_end)
        raise StopIteration


PositionalWord = namedtuple('PositionalWord', ['text', 'start', 'end'])


def positional_splitter(regex, text):
    r"""Yield sentence chunks (as defined by the regex) as well as their location.

    NOTE: the regex needs to be an "inverse match"
    Example:
        To split on whitespace, you match:
        r'\S+'  <-- "a chain of anything that's NOT whitespace"
    """
    for res in re.finditer(regex, text):
        yield PositionalWord(res.group(0), res.start(), res.end())
    raise StopIteration
