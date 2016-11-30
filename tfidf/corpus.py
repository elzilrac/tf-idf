#!/usr/bin/env python3

"""A Corpus is a collection of Documents. Calculates IDF and full TF-IDF.

Vocabulary:
    stem - to stem (or "stemming") is to conver to a word from the original
        conjugated, plural, etc to a simpler form.
            Example:
                cats -> cats
                winningly -> win
    term - a one or multi word string, no stemming
    ngram - a one or multi word string that has been pre-processed and stemmed.
        Stemming may not be idempotent, so it's important to not re-process the
        ngram.
        bi-gram = two-word ngram
        tri-gram = three-word ngram
        n-gram = "n" word ngram
    keyword - contains the ngram, references to all the document(s) it is located
        in, and the locations within the document so that the original term can
        also be re-calculated.
    document - a body of text
    corpus - a collection of documents that are at least somewhat comparible.
        it is recommended to NOT have multiple languages within a single corpus.
    TF - Term Frequency within a document
    IDF - Inverse Document Frequency, the inverse frequency of the term across
        all documents in a corpus
    TF-IDF - a combination of the TF and IDF scores reflecting the "importance"
        of a term within a particular document
"""

from __future__ import absolute_import, division

import math
from collections import namedtuple

from .document import Document
from .preprocess import Preprocessor, clean_text

Keyword = namedtuple('Keyword', ['term', 'ngram', 'score'])


class Corpus(object):
    """A corpus is made up of Documents, and performs TF-IDF calculations on them.

    After initialization, add "documents" to the Corpus by adding text
    strings with a document "key". These will generate Document objects.

    Example:
        >>> c = Corpus(gramsize=2)
        >>> c['doc1'] = 'Mary had a little lamb'
    """

    def __init__(self, gramsize=1, all_ngrams=True, language=None, preprocessor=None):
        """Initalize.

        Parameters:
            gramsize (int): number of words in a keyword
            all_ngrams (bool):
                if True, return all possible ngrams of size "gramsize" and smaller.
                else, only return keywords of exactly length "gramsize"
            language (str):
                Uses NLTK's stemmer and local stopwords files appropriately for the
                language.
                Check available languages with Preprocessor.supported_languages
            preprocessor (object):
                pass in a preprocessor object if you want to manually configure stemmer
                and stopwords
        """
        self.__documents = {}
        self.__gramsize = gramsize
        if preprocessor:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = Preprocessor(
                language=language, gramsize=gramsize, all_ngrams=all_ngrams)

    def __len__(self):
        """Length of a Corpus is the number of Documents it holds."""
        return len(self.__documents)

    def __getitem__(self, document_id):
        """Fetch a Document from the Corpus by its id."""
        return self.__documents[document_id]

    def __setitem__(self, document_id, text):
        """Add a Document to the Corpus using a unique id key."""
        text = clean_text(text)
        self.__documents[document_id] = Document(text, self.preprocessor)

    @property
    def gramsize(self):
        """Number of words in the ngram. Not editable post init."""
        return self.__gramsize

    def keys(self):
        """The document ids in the corpus."""
        return self.__documents.keys()

    @property
    def max_raw_frequency(self):
        """Highest frequency across all Documents in the Corpus."""
        return max([_.max_raw_frequency for _ in self.__documents.values()])

    def count_doc_occurances(self, ngram):
        """Count the number of documents the corpus has with the matching ngram."""
        return sum([1 if ngram in doc else 0 for doc in self.__documents.values()])

    def idf_basic(self, ngram):
        if self.count_doc_occurances(ngram) == 0:
            raise Exception(ngram)
        return math.log(float(len(self)) / self.count_doc_occurances(ngram))

    def idf_smooth(self, ngram):
        return math.log(1 + (float(len(self)) / self.count_doc_occurances(ngram)))

    def idf_max(self, ngram):
        return math.log(1 + self.max_raw_frequency / self.count_doc_occurances(ngram))

    def idf_probabilistic(self, ngram):
        num_doc_occurances = self.count_doc_occurances(ngram)
        return math.log(float(len(self) - num_doc_occurances) / num_doc_occurances)

    def idf(self, ngram, idf_weight='smooth'):
        """Inverse document frequency (IDF) indicates ngram common-ness across the Corpus."""
        if idf_weight == 'smooth':
            return self.idf_smooth(ngram)
        if idf_weight == 'basic':
            return self.idf_basic(ngram)
        if idf_weight == 'max':
            return self.idf_max(ngram)
        if idf_weight == 'prob':
            return self.idf_probabilistic(ngram)

    def tf_idf(self, term, document_id=None, text=None, idf_weight='basic', tf_weight='basic',
               normalize_term=True):
        """TF-IDF score. Must specify a document id (within corpus) or pass text body."""
        assert document_id or text
        document = None
        if document_id:
            document = self[document_id]
        if text:
            text = clean_text(text)
            document = Document(text, self.preprocessor)
        if normalize_term:
            ngram = self.preprocessor.normalize_term(term)
        else:
            ngram = term
        score = document.tf(ngram, tf_weight=tf_weight) * self.idf(ngram, idf_weight=idf_weight)
        return Keyword(term, ngram, score)

    def get_keywords(self, document_id=None, text=None, idf_weight='basic',
                     tf_weight='basic', limit=100):
        """Return a list of keywords with TF-IDF scores. Defaults to the top 100."""
        assert document_id or text
        document = None
        if document_id:
            document = self[document_id]
        if text:
            text = clean_text(text)
            document = Document(text, self.preprocessor)
        out = []
        for ngram, kw in document.keywordset.items():
            score = document.tf(ngram, tf_weight=tf_weight) * \
                self.idf(ngram, idf_weight=idf_weight)
            out.append(Keyword(kw.get_first_text(), ngram, score))
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:limit]
