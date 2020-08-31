import os
import tarfile
import tempfile
import pickle
import shutil
import numpy as np

from affine.detection.model.classifiers import AbstractClassifier, LibsvmClassifier
from affine.detection.model.features import SurfExtractor, SpatialBagOfWords
from affine.detection.model.projections import PCAScikit
from affine.detection.model import cross_validation
from ..utils.scene_functions import POS_LABEL, NEG_LABEL, IM_WIDTH, IM_HEIGHT, get_config

__all__ = ['SpatialSceneClassifier']

# default is no PCA
CFG_SPEC = """
    [classification]
    pca_dimensions = integer(default=0)
    vocab_size = integer(default=500)
    keypoint_intv = integer(min=1, default=15)
    svm_type = integer(default=3)
    svm_threshold = float(default=0)
    svm_kernel = integer(default=5)
    svm_nu = float(default=0.2)
    svm_gamma = float(default=0.05)
    num_levels = integer(min=1, default=3)
    """

class SpatialSceneClassifier(AbstractClassifier):
    model_names = ['sbow', 'svm', 'params']

    def __init__(self, config_file):
        self.config_file = config_file
        config = get_config(self.config_file, CFG_SPEC.split('\n'))
        self._trained = False
        self.params = config['classification']
        self.svm = LibsvmClassifier()
        self.svm.set_svm_params(self._extract_svm_params(self.params))
        feat_ext = SurfExtractor(keyp_grid_params=[IM_WIDTH, IM_HEIGHT,
                                                   self.params['keypoint_intv']])
        pca = None
        pca_dims = self.params['pca_dimensions']
        if pca_dims > 0:
            pca = PCAScikit(n_components=pca_dims)
        self.sbow = SpatialBagOfWords(self.params['num_levels'], IM_HEIGHT,
                                      IM_WIDTH, feat_ext,
                                      vocabsize=self.params['vocab_size'],
                                      projection=pca)

    def save_to_dir(self, model_dir):
        """Saves models to directory.

        Creates directory if it does not exist.

        Args:
            model_dir: Path to directory.

        Raises:
            AssertionError: Classifier is not trained.
        """
        assert self._trained, 'Classifier is not trained'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open(self._params_path(model_dir), 'w') as f:
            pickle.dump(self.params, f)
        self.sbow.save_to_dir(self._sbow_path(model_dir))
        self.svm.save_to_file(self._svm_path(model_dir))

    @staticmethod
    def load_from_dir(model_dir):
        """Loads model directory.

        Args:
            model_dir: Path to directory.

        Returns:
            A classifier.
        """
        cls = SpatialSceneClassifier
        clf = cls.__new__(cls)
        with open(clf._params_path(model_dir), 'r') as f:
            clf.params = pickle.load(f)
        clf.sbow = SpatialBagOfWords.load_from_dir(clf._sbow_path(model_dir))
        clf.svm = LibsvmClassifier.load_from_file(clf._svm_path(model_dir))
        clf._trained = True
        return clf

    def train(self, images, labels):
        """Train classifier.

        Args:
            images: List of image paths.
            labels: Corresponding list of labels.

        Raises:
            AssertionError: Labels are not +/- 1.
        """
        assert all([l in [POS_LABEL, NEG_LABEL] for l in labels]), 'Invalid labels'
        _, hists = self.sbow.train(images)
        self.svm.train(hists, np.asarray(labels))
        self._trained = True

    def test(self, images, svm_threshold=None):
        """Test classifier.

        Args:
            images: List of image paths.
            svm_threshold: Optional SVR decision threshold

        Returns:
            List of labels.

        Raises:
            AssertionError: Classifier is not trained.
        """
        assert self._trained, 'Classifier is not trained'
        _, hists = self.sbow.extract(images)
        results = self.svm.predict(hists).tolist()
        if self._uses_svr:
            if svm_threshold is None:
                svm_threshold = self.image_threshold
            results = [POS_LABEL if r >= svm_threshold else NEG_LABEL for r in results]
        return results

    def evaluate(self, images, labels, n_folds=5):
        """Evaluate classifier using cross-validation.

        Results may be meaningless if there are too few images.

        Args:
            images: List of image paths.
            labels: Corresponding list of labels.

        Returns:
            A tuple consisting of precision and recall averaged over all folds.
        """
        metrics = cross_validation.std_cross_validation(self, np.asarray(images),
                                                        np.asarray(labels),
                                                        n_folds=n_folds)
        precision = np.mean(metrics['precision'])
        recall = np.mean(metrics['recall'])
        return precision, recall

    @property
    def image_threshold(self):
        return self.params['svm_threshold'] if self._uses_svr else None

    def __deepcopy__(self, _):
        cls = SpatialSceneClassifier
        if self._trained:
            tmp_dir = tempfile.mkdtemp()
            self.save_to_dir(tmp_dir)
            clf = cls.load_from_dir(tmp_dir)
            shutil.rmtree(tmp_dir)
            return clf
        return cls(self.config_file)

    @staticmethod
    def _extract_svm_params(params):
        return {'svm_type': params['svm_type'], 'kernel_type': params['svm_kernel'],
                'nu': params['svm_nu'], 'gamma': params['svm_gamma']}

    @property
    def _uses_svr(self):
        return self.params['svm_type'] in [3, 4]

    @staticmethod
    def _sbow_path(d):
        return os.path.join(d, 'sbow')
    
    @staticmethod
    def _svm_path(d):
        return os.path.join(d, 'svm')

    @staticmethod
    def _params_path(d):
        return os.path.join(d, 'params')

