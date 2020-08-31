import re

pp_re = re.compile(r'http|\W|\d|_')


def url_analyzer(url):
    tokens = []
    tt = url.split('?', 1)[0]
    tt = ''.join(pp_re.split(tt))
    ngrams = _char_ngrams(tt)
    return ngrams


def _char_ngrams(text, nrange=(4, 8)):
    text = text.lower()
    ngrams = []
    min_n, max_n = nrange
    text_len = len(text)
    for n in xrange(min_n, min(max_n + 1, text_len + 1)):
        for i in xrange(text_len - n + 1):
            ngrams.append(text[i: i + n])
    return ngrams
