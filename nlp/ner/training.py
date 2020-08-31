from logging import getLogger
import argparse
import os

import numpy as np

from affine.detection.model.features import NerFeatureExtractor
from affine.detection.model.classifiers import LibsvmClassifier
from affine.detection.model import cross_validation
from affine.model import Label, WebPage


logger = getLogger(__name__)

class NerTrainer(object):
    FTR_COLS = ['pid', 'split1', 'split2', 'split3', 'title', 'cat_uniq']
    TF_FILE = 'true_feature_file'
    FF_FILE = 'false_feature_file'
    MODEL_FILE = 'model.svm'

    def __init__(self, label_id):
        self.label_id = label_id

    def create_ftr_file(self, pid_file, ftr_file):
        nfe = NerFeatureExtractor()
        page_ids = np.loadtxt(pid_file, usecols=[0], dtype=int)
        with open(ftr_file, 'w') as fo:
            fo.write('\t'.join(self.FTR_COLS)+'\n')
            for pid in page_ids:
                fv = nfe.entity_featurize(WebPage.get(pid), self.label_id)
                if sum(fv):
                    fv = np.hstack((pid, fv))  # first col is page_id
                    fo.write('\t'.join([str(i) for i in fv]) + '\n')

    @classmethod
    def read_ftr_files(cls, true_ftr_file, false_ftr_file, remove_outliers=False):
        true_x = np.loadtxt(true_ftr_file, skiprows=1, usecols=range(1, len(cls.FTR_COLS)), dtype=int)
        false_x = np.loadtxt(false_ftr_file, skiprows=1, usecols=range(1, len(cls.FTR_COLS)), dtype=int)
        if remove_outliers:
            true_x = cls.filter_outliers(true_x, ignore_ftrs=[cls.FTR_COLS.index('title') - 1])
            false_x = cls.filter_outliers(false_x, ignore_ftrs=[cls.FTR_COLS.index('title') - 1])
        xx = np.vstack((true_x, false_x))
        yy = np.hstack((np.ones(true_x.shape[0]), np.zeros(false_x.shape[0])))
        return xx, yy

    @classmethod
    def filter_outliers(cls, xx, ignore_ftrs=[]):
        ''' Taken from http://en.wikipedia.org/wiki/Quartile#Outliers '''
        lower_quartiles = np.percentile(xx, 25, axis=0)
        upper_quartiles = np.percentile(xx, 75, axis=0)
        iqr = upper_quartiles - lower_quartiles
        logical_index = np.ones(xx.shape[0], dtype=bool)
        for i in set(range(xx.shape[1])) - set(ignore_ftrs):
            logical_index = logical_index & (xx[:, i] > lower_quartiles[i] - 1.5*iqr[i])\
                    & (xx[:, i] < upper_quartiles[i] + 1.5*iqr[i])
        return xx[logical_index]

    def run_pipeline(self, args):
        true_ftr_file = os.path.join(args.op_dir, self.TF_FILE)
        false_ftr_file = os.path.join(args.op_dir, self.FF_FILE)
        logger.info('Creating feature files')
        self.create_ftr_file(args.true_pid_file, true_ftr_file)
        self.create_ftr_file(args.false_pid_file, false_ftr_file)
        logger.info('Training classifier')
        self.train_classifier(true_ftr_file, false_ftr_file, args.op_dir)
        logger.info('Model files written to %s'%args.op_dir)

    def train_classifier(self, true_ftr_file, false_ftr_file, op_dir, clf=LibsvmClassifier()):
        xx, yy = self.read_ftr_files(true_ftr_file, false_ftr_file)
        cross_validation.std_cross_validation(clf, xx, yy)
        clf.train(xx, yy)
        clf.save_to_file(os.path.join(op_dir, self.MODEL_FILE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_id', help='Label id')
    parser.add_argument('true_pid_file', help='File with one True page id per line')
    parser.add_argument('false_pid_file', help='File with one False page id per line')
    parser.add_argument('op_dir', help='Directory to put trained models in')
    args = parser.parse_args()
    assert Label.get(args.label_id), 'Invalid label id'
    trainer = NerTrainer(args.label_id)
    trainer.run_pipeline(args)


if __name__ == '__main__':
    main()
