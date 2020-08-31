from concurrent.futures import Future
import nltk
from nltk.corpus import movie_reviews
import random
from Stemmer import Stemmer

from affine.detection.model.flow import Step, Flow, IterFlow, \
    FutureFlowInput, FutureLambda


class CorpusSupplier(object):

    @classmethod
    def get_movie_reviews(cls):
        documents = [(category, movie_reviews.raw(fileid))
                     for category in movie_reviews.categories()
                     for fileid in movie_reviews.fileids(category)]
        random.shuffle(documents)
        l, d = zip(*documents)
        return l, d


class StopwordFilter(object):

    def __init__(self, stop_file):
        with open(stop_file) as fi:
            self.stop_words = set(fi.read().decode('utf-8').splitlines())

    def remove_stopwords(self, text):
        clean = [w for w in nltk.word_tokenize(
            text) if w not in self.stop_words]
        return clean


class MyStemmer(object):

    def __init__(self, stemmer_type):
        self.stemmer = Stemmer(stemmer_type)

    def do_stemming(self, word_list):
        return self.stemmer.stemWords(word_list)


class FrequentWordFeatures(object):

    def __init__(self, N=2000):
        self.N = N
        self.ftr_words = []

    def feature_setter(self, docs):
        freq = nltk.FreqDist([w for ww in docs for w in ww])
        self.ftr_words = freq.keys()[:self.N]

    def featurize(self, word_list):
        assert len(self.ftr_words) == self.N
        bow = set(word_list)
        fv = {}
        for w in self.ftr_words:
            fv[w] = w in bow
        return fv

stopwords_file = "/home/hardik/nltk_data/corpora/stopwords/english"


def main1():
    labels, corpus = CorpusSupplier.get_movie_reviews()
    sf = StopwordFilter(stopwords_file)
    stemmer = MyStemmer('english')
    docs = []
    for idx, ll in enumerate(corpus):
        pp = sf.remove_stopwords(ll)
        docs.append(stemmer.do_stemming(pp))
    fe = FrequentWordFeatures()
    fe.feature_setter(docs)
    fv = []
    for ll in docs:
        fv.append(fe.featurize(ll))

    featuresets = zip(fv, labels)
    train_set, test_set = featuresets[100:], featuresets[:100]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))


def split_data(features, labels):
    featuresets = zip(features, labels)
    train_set, test_set = featuresets[100:], featuresets[:100]
    return train_set, test_set


def nltk_train(featuresets):
    classifier = nltk.NaiveBayesClassifier.train(featuresets)
    return classifier


def nltk_test(classifier, test_set):
    print nltk.classify.accuracy(classifier, test_set)


def main2():
    logger.l
    labels, corpus = CorpusSupplier.get_movie_reviews()
    f = Flow('data_processor')

    stemmer = Step('Stem', MyStemmer('english'), 'do_stemming')
    stopword_filter = Step(
        'stop_word_filter', StopwordFilter(stopwords_file), 'remove_stopwords')

    for step in [stemmer, stopword_filter]:
        f.add_step(step)

    f.start_with(stopword_filter, FutureFlowInput(f, 'ip_data'))
    f.connect(stopword_filter, stemmer, stopword_filter.output)
    f.output = stemmer.output
    # Setting up iterative data processing
    iter_flow = IterFlow(f)

    fwf = FrequentWordFeatures()
    # Setting up iterative feature Extraction
    f2 = Flow(name='feature Extraction')
    extract = Step('featurize', fwf, 'featurize')
    f2.add_step(extract)
    f2.start_with(extract, FutureFlowInput(f2, 'ip_data'))
    f2.output = extract.output
    iter_flow2 = IterFlow(f2)

    # Setting up entire pipeline
    f3 = Flow(name="movie-sentiments")
    data_proc = Step("data processing", iter_flow, 'operate')
    featurizer_init = Step("Feature Extraction", fwf, 'feature_setter')
    extraction = Step('extraction', iter_flow2, 'operate')
    splitter = Step("Split", split_data, None)
    classifier = Step('training', nltk_train, None)
    tester = Step("Testing", nltk_test, None)

    for step in [data_proc, featurizer_init, extraction, splitter, classifier, tester]:
        f3.add_step(step)

    f3.start_with(data_proc, FutureFlowInput(f3, 'corpus'))
    f3.connect(data_proc, featurizer_init, data_proc.output)
    f3.connect(featurizer_init, extraction, data_proc.output)
    f3.connect(extraction, splitter, extraction.output,
               FutureFlowInput(f3, 'labels'))
    f3.connect(splitter, classifier, FutureLambda(
        splitter.output, lambda x: x[0]))
    f3.connect([classifier, splitter], tester, classifier.output,
               FutureLambda(splitter.output, lambda x: x[1]))

    return f, f2, f3


if __name__ == '__main__':
    main1()
