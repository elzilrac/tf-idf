#!/usr/bin/env python3

"""The basic unit of TF-IDF, the keyword.

This class allows you to have stemmed keywords, but still see the original text.
"""

from collections import namedtuple

Location = namedtuple('Location', ['document', 'start', 'end'])


class Keyword(object):
    def __init__(self, text, document=None, start=None, end=None):
        self.locations = set()
        self.text = text
        self.locations = set()
        if (start is not None) and (end is not None):
            self.locations.add(Location(document, start, end))

    def update_locations(self, locations):
        self.locations = self.locations.union(locations)

    def __add__(self, other):
        assert self.text == other.text
        out = Keyword(self.text)
        out.locations = self.locations
        out.update_locations(other.locations)
        return out

    def __ladd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __len__(self):
        return len(self.locations)

    @property
    def original_texts(self):
        out = []
        for loc in self.locations:
            if loc.document:
                text = loc.document.text[loc.start:loc.end]
            else:
                text = ''
            out.append(text)
        return list(set(out))

    def get_first_text(self):
        """Return the first original text."""
        loc = next(iter(self.locations))
        return loc.document.text[loc.start:loc.end]

    def __str__(self):
        return 'Stem:%s, Instances:%s, Count:%d' % (self.text, str(self.original_texts), len(self))
