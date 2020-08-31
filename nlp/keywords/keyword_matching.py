from collections import defaultdict
from unidecode import unidecode
import re

import nltk
from Stemmer import Stemmer

__all__ = ['process_text', 'PageKeywordMatcher', 'PageEntityMatcher']

clean_re = re.compile(r"[\W_]", re.UNICODE)
word_tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()


def _tokenize(text):
    if isinstance(text, str):
        text = text.decode('utf-8')
    words = word_tokenizer.tokenize(text.lower())
    words = [clean_re.sub("", word) for word in words]
    return filter(None, words)


def process_text(text, stemming=True):
    words = _tokenize(text)
    if not stemming:
        return words
    stemmer = Stemmer('english')
    return [stemmer.stemWord(word) for word in words]


class PageKeywordMatcher(object):
    def __init__(self):
        self.keywords = defaultdict(list)
        self.longest_keyword = 0

    def _process(self, text):
        if isinstance(text, basestring):
            text = process_text(text)
        return text or []

    def add_keyword(self, kw_id, kw_text):
        kw_text = tuple(self._process(kw_text))
        if kw_text:
            self.keywords[kw_text].append(kw_id)
        self.longest_keyword = max(len(kw_text), self.longest_keyword)

    def matching_keywords(self, text):
        text = self._process(text)
        matches = set()
        for start in xrange(len(text)):
            for kw_len in xrange(1, self.longest_keyword + 1):
                kw_text = tuple(text[start: start + kw_len])
                matches.update(self.keywords[kw_text])
        return matches


class PageEntityMatcher(object):
    def __init__(self):
        self.entities = {}
        self.longest_entity = 0

    def _process(self, text):
        if isinstance(text, basestring):
            text = process_text(text, stemming=False)
        text = [unidecode(w) for w in text]
        return text or []

    def add_entity(self, ne_id, ne_text):
        ne_text = tuple(self._process(ne_text))
        if ne_text:
            self.entities[ne_text] = ne_id
        self.longest_entity = max(len(ne_text), self.longest_entity)

    def matching_entities(self, text):
        text = self._process(text)
        matches = []
        env = -1
        for start in xrange(len(text)):
            for ne_len in xrange(self.longest_entity, 0, -1):
                if min(start + ne_len, len(text)) <= env:
                    break
                ne_text = tuple(text[start: start + ne_len])
                ne_id = self.entities.get(ne_text)
                if ne_id is not None:
                    matches.append(ne_id)
                    env = min(start + ne_len, len(text))
        return matches
