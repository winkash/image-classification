import os
import pickle
import random
import shutil
from configobj import ConfigObj
from collections import defaultdict
from logging import getLogger
from tempfile import mkdtemp
from validate import Validator

import numpy as np

from affine.model import *
from affine.detection.model.projections import *
from affine.detection.vision.facerec import *
from affine.detection.vision.facerec.data_acquisition_utils import get_data, get_negative_data


__all__ = ['FaceIndexBuilder']

logger = getLogger('affine.face_index_builder')

PCA_FILE_NAME = 'pca.xml'
LDA_FILE_NAME = 'lda.pickle'
DLA_FILE_NAME = 'dla.pickle'
HASH_TABLE_FILE_NAME = 'hash_tables.pickle'
FACE_INDEX_FILE_NAME = 'face_index.pickle'
BUILDER_FILE_NAME = 'face_index_builder.pickle'

CFG_SPEC = """
    base_dir = string(default='.')
    label_ids = string(default='all')
    train_count = integer(default='10000')
    train_ratio = float(default='0.8')
    neg_count = integer(default='5000')
    min_confidence = float(default='0.0')
    npca = integer(default='400')
    k1 = integer(default='5')
    k2 = integer(default='10')
    d = integer(default='150')
    beta = float(default='0.5')
    t = integer(default='10000')
    num_tables = integer(default='30')
    num_clusters = integer(default='600')"""


class FaceIndexBuilder(object):
    def __init__(self, config_file):
        config_dict = self.validate_config_file(config_file)
        # all params are being set as attributes for the builder
        for k, v in config_dict.items():
            setattr(self, k, v)

        if self.label_ids != 'all':
            self.label_ids = map(int, self.label_ids.split(','))
        self.pca = PCA(self.npca)
        self.dla = DLA(self.k1, self.k2, self.d, self.beta)

        if not os.path.isdir(self.base_dir):
            os.makedirs(self.base_dir)
        logger.info('Face Index Builder using base_dir : %s' %self.base_dir)

    def validate_config_file(self, config_file):
        """Validate the given config file and raise error if validation fails"""
        config_dict = ConfigObj(config_file, configspec=CFG_SPEC.split('\n'))
        result = config_dict.validate(Validator(), copy=True, preserve_errors=True)
        if result != True:
            msg = 'Config file validation failed: %s' % result
            raise Exception(msg)
        return config_dict

    def get_negative_data(self):
        self.neg_bag, _ = get_negative_data(self.neg_count, ratio=1.0, min_confidence=self.min_confidence, single_box_per_video=False)

    def get_data(self):
        # since in DLA, we need k1 same class members for each point,
        # each class needs atleast k1+1 points
        min_boxes_required = self.k1 + 1
        self.train_bag, self.test_bag = get_data(self.label_ids, self.train_count, self.train_ratio,
                                                 min_boxes=min_boxes_required,
                                                 min_confidence=self.min_confidence,
                                                 single_box_per_video=False)

    def pickle_dump(self, obj, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    def learn_dla_sig_extractor(self):
        # Collect features only if files are not given
        logger.info('Starting to build Signature Extractor')
        logger.info('Obtaining face signatures from DB')
        train_feats = self.train_bag.feats

        # Computing PCA
        logger.info('Computing PCA projection matrices')
        pca_file = self.pca.train(train_feats)
        # copy to base dir
        pca_path = os.path.join(self.base_dir, PCA_FILE_NAME)
        shutil.copy(pca_file, pca_path)
        logger.info('PCA xml file stored at: %s' %pca_path)

        # Computing DLA
        logger.info('Computing new DLA model')
        pca_projected_feats = self.pca.project(train_feats)
        self.dla.train(pca_projected_feats, self.train_bag.labs)
        dla_file = os.path.join(self.base_dir, DLA_FILE_NAME)
        self.dla.save_to_file(dla_file)
        logger.info('DLA model stored at: %s' %dla_file)

        self.sig_extractor = DLAExtractor(pca_file, dla_file)

    def get_hash_tables(self):
        logger.info('Starting to get hash tables using KMeans')
        assert self.sig_extractor, "learn_sig_extractor method must be called before creating hash tables"
        projected_feats = self.sig_extractor.extract(self.train_bag.box_ids)
        self.hash_tables = KMeansHashing.create_hash_tables(self.num_tables, projected_feats, self.num_clusters)
        hash_table_file = os.path.join(self.base_dir, HASH_TABLE_FILE_NAME)
        self.pickle_dump(self.hash_tables, hash_table_file)

    def build_index(self):
        logger.info('Starting to build a new Index')
        hash_size = 1
        self.learn_dla_sig_extractor()
        self.get_hash_tables()

        self.face_index = FaceIndex(hash_size, self.num_tables, self.sig_extractor, hash_type='kmeans', hash_tables=self.hash_tables)
        self.face_index.build_index(self.train_bag.box_ids, self.train_bag.labs)

        face_index_path = os.path.join(self.base_dir, FACE_INDEX_FILE_NAME)
        FaceIndex.save_to_file(self.face_index, face_index_path)

    def save_to_file(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.base_dir, BUILDER_FILE_NAME)

        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

        logger.info('Successfully saved the builder to %s' %file_path)

    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, 'rb') as f:
            fib = pickle.load(f)

        return fib

    def run_pipeline(self):
        self.get_data()
        self.get_negative_data()
        self.build_index()
