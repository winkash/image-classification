import os
import pickle

from affine.detection.model.classifiers import KNNScikit
from affine.detection.model.clustering import intersection


class KnnTrainer(object):

    """Trains k nearest neighbors. 

    Trains knn using training logos and save the knn model
    and logos to files.

    Attributes:
        bow : a bag of words object
        metric: metric for knn training
        weights: wights for knn training
        neighbors: number of neighbors for knn training
        alfgorithm: The algorithm for knn training

    """

    def __init__(self, bag_of_words, metric=intersection, weights='distance',
                 neighbors=3, algorithm='auto'):
        """Inits KnnTrainer class."""
        self.metric = metric
        self.neighbors = neighbors
        self.algorithm = algorithm
        self.weights = weights

        self.bow = bag_of_words

    def _get_logo_hists(self):
        """Gets the histogram of logos.

        Runs bag of words extraction to get histograms of logos
        and store the results in logos Attributes.

        """

        logo_paths = [l.path for l in self.logos]
        image_desc, self.logo_hists = self.bow.extract(logo_paths)
        for logo, h, img_d in zip(self.logos, self.logo_hists, image_desc):
            logo.feature = h
            logo.image_desc = img_d

    def train(self, logos):
        """Trains k nearest neighbors using training logos.
        Args: 
            logos: logo objects

        """

        self.logos = logos
        self._get_logo_hists()
        self._knnsck = KNNScikit(neighbors=self.neighbors,
                                 metric=self.metric, weights=self.weights, algorithm=self.algorithm)
        logo_target_label_ids = [l.target_label_id for l in self.logos]
        self._knnsck.train(self.logo_hists, logo_target_label_ids)

    def save_knn(self, model_dir):
        """ Saves the knn trained model.

        Args:
            model_dir: path to model directory

        """

        knn_file = self._knn_path(model_dir)
        self._knnsck.save_to_file(knn_file)

    def save_logos(self, model_dir):
        """ Saves the training logos to model directory.

        Args:
            model_dir: path to model directory
        """

        logos_file = self._logos_path(model_dir)
        with open(logos_file, 'w') as f:
            pickle.dump(self.logos, f)

    @staticmethod
    def _knn_path(model_dir):
        return os.path.join(model_dir, 'knn')

    @staticmethod
    def _logos_path(model_dir):
        return os.path.join(model_dir, 'logos')
