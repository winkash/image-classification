import math
import os
import pickle
from abc import ABCMeta, abstractmethod
from logging import getLogger
from tempfile import mkdtemp

import cv
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.lda import LDA as ScikitLDA
from sklearn.decomposition import PCA as PCA_from_scikit

from affine import config
from affine.video_processing import run_cmd
from affine.detection.model.databag import DataBag
from affine.detection.model.util import PickleMixin

__all__ = ['PCA', 'LDA', 'DLA', 'PCAScikit']

logger = getLogger('affine.detection.model.projections')


class AbstractProjection(PickleMixin):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, features, labels=None):
        """Calculate the projection matrix suing training features and labels
        if needed"""

    @abstractmethod
    def project(self, features):
        """Project features to the reduced space"""

    def project_bag(self, train_bag, test_bag):
        """Takes train and test bags, trains a model and
        returns projected bags"""
        self.train(train_bag.feats, train_bag.labs)
        projected_train = self.project(train_bag.feats)
        projected_test = self.project(test_bag.feats)

        projected_train_bag = DataBag(projected_train, train_bag.labs)
        projected_test_bag = DataBag(projected_test, test_bag.labs)

        return projected_train_bag, projected_test_bag

    def get_projected_bag(self, data_bag):
        projected_feats = self.project(data_bag.feats)
        return DataBag(projected_feats, data_bag.labs, data_bag.box_ids)

    def encode(self):
        return self.__dict__

    @classmethod
    def decode(cls, proj_dict):
        new_proj = cls.__new__(cls)
        new_proj.__dict__ = proj_dict
        return new_proj

    def train_and_project(self, features, labels=None):
        self.train(features, labels)
        return self.project(features)


class PCA(AbstractProjection):
    def __init__(self, ndims=None):
        super(PCA, self).__init__()
        self.ndims = ndims
        self.mean = None
        self.projection_mat = None

    def run_aff_pca(self, features):
        '''call aff_face_pca binary for fast PCA computation in C'''
        # we need to add an extra id in the start since
        #aff_pca_expects it that way
        ones = np.zeros((features.shape[0], 1))
        temp_feats = np.concatenate((ones, features), axis=1).\
                transpose().copy()
        temp_dir = mkdtemp()
        feature_file = os.path.join(temp_dir, 'features.xml')
        cv_mat = cv.fromarray(temp_feats)
        cv.Save(feature_file, cv_mat)

        pca_file = os.path.join(temp_dir, 'learned_pca.xml')
        bin_path = config.bin_dir()
        cmd = [
            os.path.join(bin_path, 'aff_face_pca'),
            feature_file,
            pca_file,
            '-n',
            str(self.ndims)
        ]
        run_cmd(cmd)
        return pca_file

    def load_from_xml_file(self, pca_file):
        self.projection_mat = np.asarray(cv.Load(pca_file,
            name='WP')).transpose()
        self.ndims = self.projection_mat.shape[1]
        self.mean = np.asarray(cv.Load(pca_file, name='CP')).transpose()

    def train(self, features, *args):
        assert self.ndims, "ndims need to be provided for training a PCA model"
        if self.mean is None or self.projection_mat is None:
            pca_file = self.run_aff_pca(features)
            self.load_from_xml_file(pca_file)
            return pca_file

    def project(self, features):
        ''' feature_mat: 1937xN dim matrix i.e. each row is a feature'''
        mean_features = features - self.mean
        return np.dot(mean_features, self.projection_mat)


class PCAScikit(AbstractProjection):
    """ A wrapper around sklearn's PCA class
    (http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    """
    def __init__(self, **kwargs):
        self.pca = PCA_from_scikit(**kwargs)

    def train(self, features):
        self.pca.fit(features)

    def project(self, features):
        return self.pca.transform(features)


class LDA(AbstractProjection):
    def __init__(self, **kw):
        super(LDA, self).__init__()
        self.lda = ScikitLDA(**kw)

    def train(self, features, labels):
        red_feats = self.lda.fit_transform(features, labels)
        self.V = np.std(red_feats, axis=0)

    def project(self, feats, whiten=True):
        lda_feats = self.lda.transform(feats)
        if whiten:
            lda_feats /= self.V
        return lda_feats


class DLA(AbstractProjection):
    """ Discriminative Locality Alignment
    Implementation of the DLA Algorithm as mentioned in
    "http://www.rad.upenn.edu/sbia/Tianhao.Zhang/
    Discriminative%20Locality%20Alignment.pdf"
    All varibales used match the same as mentioned in the above paper
    """
    def __init__(self, k1, k2, d, beta=0.5):
        super(DLA, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.d = d
        self.beta = beta
        self.dist_mat = None
        self.base_dir = mkdtemp()
        logger.info("Using base_dir : %s for storing numpy.memmaps" \
                % self.base_dir)

    def create_memmap(self, name, shape, dtype='float64', mode='w+'):
        path = os.path.join(self.base_dir, name)
        mat = np.memmap(path, dtype=dtype, mode=mode, shape=shape)
        return mat

    def get_dist_row(self, row_id):
        row = []
        # function to get distance between indices i, j in a
        # condensed distance matrix
        # having n feature points

        def get_idx(i, j, n):
            if i == j:
                return -1
            elif i > j:
                return n * j - j * (j + 1) / 2 + i - 1 - j
            else:
                return n * i - i * (i + 1) / 2 + j - 1 - i

        for idx in range(self.n_feats):
            if get_idx(row_id, idx, self.n_feats) == -1:
                row.append(0)
            else:
                row.append(self.dist_mat[get_idx(row_id, idx, self.n_feats)])
        return row

    def compute_distance_matrix(self):
        logger.info("Computing Distance matrix")
        if self.dist_mat is None:
            shape = (1, self.n_feats * (self.n_feats - 1) / 2)
            self.dist_mat = self.create_memmap('dist_mat.dat', shape)
            self.dist_mat = pdist(self.feats)

    def build_neighbors(self):
        """Create the neighbor and the weight dictionaries by
        computing nearest neighbors
        from same and differnt classes respectively
        """
        self.compute_distance_matrix()
        logger.info("Getting neighbors")
        #self.weights = {}
        self.neighbors = {}
        self.Fi = {}
        for row_id, row_label in enumerate(self.labels):
            # Make getting neighbors an independent function that
            # can be pool-mapped
            row = self.get_dist_row(row_id)
            row_neighbors = [(i, d, l) for i, (d, l) \
                    in enumerate(zip(row, self.labels))]
            row_neighbors.sort(key=lambda x: x[1])

            same_class_neighbors = []
            diff_class_neighbors = []
            for idx, dist, label in row_neighbors[1:]:
                if label == row_label and len(same_class_neighbors) < self.k1:
                    same_class_neighbors.append((idx, dist, label))
                if label != row_label and len(diff_class_neighbors) < self.k2:
                    diff_class_neighbors.append((idx, dist, label))
                if len(same_class_neighbors) == self.k1 and \
                        len(diff_class_neighbors) == self.k2:
                    break

            # confirm that we have the minimum needed neighbors
            if len(same_class_neighbors) < self.k1:
                raise ValueError("Found only %d same class neighbors for " \
                        % len(same_class_neighbors) +\
                        "label_id %d, need atleast %d" % (row_label, self.k1))
            if len(diff_class_neighbors) < self.k2:
                raise ValueError("Found only %d diff class neighbors " \
                        % len(diff_class_neighbors) +\
                        "for label_id %d, need atleast %d"
                                 % (row_label, self.k2))

            self.neighbors[row_id] = same_class_neighbors +\
                    diff_class_neighbors
            self.Fi[row_id] = [i for i, d, l in self.neighbors[row_id]]

    def create_Li(self):
        # w = [1, 1, ... -beta, -beta]
        w = np.ones(self.k1 + self.k2)
        w[self.k1:] *= -self.beta

        dim = self.k1 + self.k2 + 1
        self.Li = np.zeros((dim, dim))
        self.Li[1:, 1:] = np.diag(w)
        self.Li[0, 0] = self.k1 - self.beta * self.k2
        self.Li[0, 1:] = -w
        self.Li[1:, 0] = -w

    def compute_alignment_matrix(self):
        """Compute the alignment matrix L
        using L(Fi, Fi) <-- L(Fi, Fi) + mi*Li
        """
        logger.info("computing alignment matrix")
        self.create_Li()
        shape = (self.n_feats, self.n_feats)
        self.L = self.create_memmap('l_mat.dat', shape)
        for row_id, neighbor_idx in self.Fi.iteritems():
            indices = [row_id] + neighbor_idx
            rows = np.asarray([indices] * (self.k1 + self.k2 + 1))
            cols = rows.T
            # if using weights then use this equation
            # self.L[rows, cols] += self.weights[row_id]*self.Li
            self.L[rows, cols] += self.Li

    def learn_projection_matrix(self):
        """Learn the projection matrix U to project to 'd'
        dimensional subspoace
        """
        self.build_neighbors()
        self.compute_alignment_matrix()
        # getting 'd' smallest eigenvectors for XT*U*X
        logger.info("Performing eigen value decomposition")

        shape = (self.feats.T.shape[0], self.L.shape[1])
        temp = self.create_memmap('temp_mat.dat', shape)
        temp = np.dot(self.feats.T, self.L)

        shape = (temp.shape[0], self.feats.shape[1])
        mat = self.create_memmap('mat.dat', shape)
        mat = np.dot(temp, self.feats)

        eig_vals, eig_vecs = np.linalg.eig(mat)
        idx_order = np.argsort(eig_vals)
        self.U = eig_vecs[:, idx_order[:self.d]]

    def train(self, feats, labels):
        """ feats : feature mat, N x m (each row is a data point)
            labels : list of length N
        """
        assert feats.shape[0] == len(labels),\
        "feats and labels do not match in dimensions"
        self.feats = self.create_memmap('feats.dat', feats.shape)
        self.feats = feats
        self.labels = labels
        self.n_feats = len(self.feats)
        self.learn_projection_matrix()

    def project(self, feats):
        """ feats : each row represents a point (N x m)
            returns : projected features (N x d) where d < m
        """
        assert self.U is not None, "Projection matrix U missing"
        return np.dot(feats, self.U)

    def save_to_file(self, file_path):
        assert self.U is not None, "Projection matrix U missing"
        dla_dict = self.encode()
        with open(file_path, 'w') as f:
            pickle.dump(dla_dict, f)

    def encode(self):
        dla_dict = {
            'k1': self.k1,
            'k2': self.k2,
            'd': self.d,
            'beta': self.beta,
            'U': self.U
        }
        return dla_dict

    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, 'r') as f:
            dla_dict = pickle.load(f)
            return cls.decode(dla_dict)
