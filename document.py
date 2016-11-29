#!/usr/bin/env python3

"""A document object used for the "term frequency" (TF) in TF-IDF.

Example:
    >>> input_text = '''Mary had a little lamb, his fur was white as snow
            Everywhere the child went, the lamb, the lamb was sure to go'''
    >>> d = Document(input_text)
    >>> d.max_raw_frequency('lamb')
    3

    >>> d.tf_raw('lamb')
    0.13043478260869565

"""

# from collections import namedtuple
import math
import random
# import re
from .preprocess import clean_text
# from .porter2 import stem
# from .keyword import Keyword


class Document(object):
    """A document holds text, slices text into ngrams, and calculates tf score.

    Attributes:
        stop_words (list): TODO
        text (list): cleaned text, set on init
        gram_breaks (regex): if a word ends with one of these characters, an
            ngram may not cross that. Example:
            in the sentence "Although he saw the car, he ran across the street"
            "car he" may not be a bi-gram
    """

    def __init__(self, raw_text, preprocessor):
        """All you need is the text body and gramsize (number words in ngram).

        raw_text
            text string input. Will be run through text preprocessing
        preprocessor
            initalized instance of a preprocessor
        """
        self.id = None
        self.text = clean_text(raw_text)
        self.__keywordset = None
        self.__max_raw_frequency = None
        self.__length = None
        self.preprocessor = preprocessor

    def __contains__(self, ngram):
        """Check if the ngram is present in the document."""
        return ngram in self.keywordset

    def __getitem__(self, ngram):
        """Return the Keyword object with occurances via the stemmed ngram."""
        return self.keywordset[ngram]

    def __len__(self):
        """The length of the document is the number of ngrams."""
        if not self.__length:
            self.__length = sum([len(x) for x in self.keywordset])
        return self.__length

    @property
    def max_raw_frequency(self):
        """Max ngram frequency found in document."""
        if not self.__max_raw_frequency:
            biggest_kw = ''
            for kw in self.keywordset:
                if len(kw) > len(biggest_kw):
                    biggest_kw = kw
            self.__max_raw_frequency = len(biggest_kw)
        return self.__max_raw_frequency

    @property
    def gramset(self):
        """Important for fast check if ngram in document."""
        return set([x for x in self.keywordset])

    @property
    def keywordset(self):
        """Return a set of keywords in the document with all their locations."""
        if not self.__keywordset:
            self.__keywordset = {}
            for kw in self.keywords:
                if kw.text not in self.__keywordset:
                    self.__keywordset[kw.text] = kw
                else:
                    self.__keywordset[kw.text] += kw
        return self.__keywordset

    # Term Frequency weighting functions:
    def tf_raw(self, ngram):
        """The frequency of an ngram in a document."""
        num_occurances = len(self[ngram]) if ngram in self else 0
        return float(num_occurances) / len(self)

    def tf_log(self, ngram):
        """The log frequency of an ngram in a document."""
        # TODO: is this formula right? Wikipedia is often wrong...
        return 1 + math.log(self.tf_raw(ngram))

    def tf_binary(self, ngram):
        """Binary term frequency (0 if not present, 1 if ngram is present)."""
        if ngram in self:
            return 1
        return 0

    def tf_norm_50(self, ngram):
        """Double normalized ngram frequency."""
        term_frequency = self.tf_raw(ngram)
        return 0.5 + (0.5 * (term_frequency / self.max_raw_frequency))

    def tf(self, ngram, tf_weight='norm_50', normalize=False):
        """Calculate term frequency.

        Normalizing (stemming) is slow, so it's assumed you're using the ngram.
        """
        if normalize:
            ngram = self.normalize_term(ngram)
        if tf_weight == 'log':
            return self.tf_log(ngram)
        if tf_weight == 'norm_50':
            return self.tf_norm_50(ngram)
        if tf_weight == 'binary':
            return self.tf_binary(ngram)
        if tf_weight == 'basic':
            return self.tf_raw(ngram)

    def randgram(self):
        """Return a random gram. Limited to first 100 ngrams."""
        return self.randkeyword().text

    def randkeyword(self):
        """Return a random keyword. Limited to first 100 ngrams."""
        rand_pos = random.randint(0, min(100, len(self.keywordset) - 1))
        for i, x in enumerate(self.keywordset.values()):
            if i == rand_pos:
                return x
        print('uh oh')
        print(rand_pos)

    @property
    def keywords(self):
        """Use the preprocessor to yield keywords from the source text."""
        return self.preprocessor.yield_keywords(self.text, document=self)
