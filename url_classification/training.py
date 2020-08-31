import argparse
import cPickle as pickle
import numpy as np
import os
import random
import sys
from collections import Counter, defaultdict
from logging import getLogger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from affine.detection.url_classification.utils import url_analyzer

logger = getLogger(__name__)


class UrlTrainer(object):

    def __init__(
            self,
            label,
            training_file,
            model_dir,
            clf=MultinomialNB(class_prior=[0.9, 0.1]),
            n_test_samples=1000):

        self.label = label
        self.training_file = training_file
        self.model_dir = model_dir
        self.clf = clf
        self.n_test_samples = n_test_samples

    def read_dataset(self):
        urls, labels = np.loadtxt(
            self.training_file, unpack=True, dtype=str,
            comments='thisisarandomstringccedx',
            usecols=(0, 1), delimiter='\t')
        return urls, labels

    def create_train_test_sets(self, urls, labels):
        ''' Splits data into training and test label, url sets'''
        assert len(urls) == len(labels)
        cc = Counter(labels)
        # set num of negative examples per label, so that training
        # set has equal number of positive and negative exmaples
        num_neg_samples = cc[self.label] / (len(cc) - 1)

        url_labels_dict = defaultdict(list)
        for u, l in zip(urls, labels):
            url_labels_dict[l].append(u)

        training_urls = []
        training_labels = []
        testing_urls = []
        testing_labels = []
        for l in url_labels_dict:
            if l == self.label:
                n_samples = cc[self.label]
            else:
                n_samples = num_neg_samples
            random.shuffle(url_labels_dict[l])
            te_urls, tr_urls = url_labels_dict[l][
                :self.n_test_samples], url_labels_dict[l][
                self.n_test_samples: self.n_test_samples + n_samples]
            if l == self.label:
                te_labels, tr_labels = [1] * len(te_urls), [1] * len(tr_urls)
            else:
                te_labels, tr_labels = [0] * len(te_urls), [0] * len(tr_urls)
            testing_labels.extend(te_labels)
            training_urls.extend(tr_urls)
            testing_urls.extend(te_urls)
            training_labels.extend(tr_labels)

        return training_urls, np.array(
            training_labels), testing_urls, np.array(testing_labels)

    @classmethod
    def url_to_features(cls, urls, labels):
        tfv = TfidfVectorizer(
            input='content',
            analyzer=url_analyzer,
            sublinear_tf=True,
            smooth_idf=True)
        array_with_zeros = tfv.fit_transform(urls)
        nz_rows = sorted(set(np.nonzero(array_with_zeros)[0]))
        training_ftrs = array_with_zeros[nz_rows]
        return tfv, training_ftrs, labels[nz_rows]

    def run_training(self):
        logger.info('Training for label %s', self.label)
        urls, labels = self.read_dataset()
        training_urls, training_labels, testing_urls, \
            testing_labels = self.create_train_test_sets(urls, labels)
        logger.info('# Training docs %d', len(training_labels))
        logger.info('# Testing docs %d', len(testing_labels))
        tfv, training_ftrs, training_labels = self.url_to_features(
            training_urls, training_labels)
        testing_ftrs = tfv.transform(testing_urls)
        self.clf.fit(training_ftrs, training_labels)
        pred_labels = self.clf.predict(testing_ftrs)
        logger.info('\n%s', classification_report(testing_labels, pred_labels))
        model_name = '%s_%s.p' % (self.label, self.clf.__class__.__name__)
        with open(os.path.join(self.model_dir, model_name), 'wb') as fo:
            pickle.dump((tfv, self.clf), fo)
        logger.info('Model files written to %s', self.model_dir)


def main(cmd_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('label', help='Training label from training file ')
    parser.add_argument('training_file',
                        help='file where each line is <URL><tab><LABEL>')
    parser.add_argument(
        'model_dir',
        help='working directory where all the model files should go.')
    # TODO: Add option for sklearn classifier
    args = parser.parse_args(args=cmd_args)
    ut = UrlTrainer(args.label, args.training_file, args.model_dir)
    ut.run_training()


if __name__ == '__main__':
    main(sys.argv[1:])
