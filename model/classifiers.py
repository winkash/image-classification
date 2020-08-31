import os
import pickle
import shutil
import cv2.cv as cv
import numpy as np
import svm

from collections import namedtuple
from abc import ABCMeta, abstractmethod
from datetime import datetime
from logging import getLogger
from tempfile import mkdtemp, mkstemp

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from affine.video_processing import run_cmd
from affine import config

__all__ = [
    'AbstractClassifier',
    'DualSubspace',
    'KNNScikit',
    'LibsvmClassifier',
    'RandomForest',
    'DecisionStump']

logger = getLogger('affine.detection.model.classifiers')


class AbstractClassifier(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._is_trained = False

    @abstractmethod
    def train(self, features, labels):
        """
        train a model based on given pair of features and corresponding labels
        """

    @abstractmethod
    def test(self, features):
        """given prodeiction results for given features"""

    def train_and_test(self, features, labels):
        self.train(features, labels)
        return self.test(features)

    @classmethod
    def load_from_file(cls, file_name):
        ''' to laod from a pickle file '''
        with open(file_name, 'rb') as f:
            return pickle.load(f)

    def save_to_file(self, file_name):
        ''' to save to a pickle file for future loads '''
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)


class DualSubspace(AbstractClassifier):

    def __init__(self, n1, n2, nmodes2, topn, thresh):
        super(DualSubspace, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.nmodes2 = nmodes2
        self.topn = topn
        self.thresh = thresh

    def face_subspace(self):
        bin_dir = config.bin_dir()
        run_cmd([os.path.join(bin_dir, 'aff_face_subspace'),
                 self.feature_file,
                 self.subspace_file,
                 '-n1', str(self.n1),
                 '-n2', str(self.n2)])

    def subspace_project(self, test_features):
        projected_file = os.path.join(self.model_dir, 'proj.xml')
        # creating copy of subspace file since we over write the same file
        temp_subspace_file = os.path.join(
            self.model_dir, 'temp_subspace_file_%s.xml' % datetime.utcnow())
        shutil.copy(self.subspace_file, temp_subspace_file)
        # create a feature file
        feature_file = os.path.join(self.model_dir, 'test_features.xml')
        la = np.asarray([[-1]*test_features.shape[0]])
        cv_feats = np.transpose(
            np.concatenate((la, test_features.transpose())))
        mat_to_save = cv_feats.transpose().copy()
        cv.Save(feature_file, cv.fromarray(mat_to_save))
        # call binary
        bin_dir = config.bin_dir()
        run_cmd([os.path.join(bin_dir, 'aff_face_subspace_project'),
                 feature_file,
                 temp_subspace_file,
                 projected_file])
        return temp_subspace_file, projected_file

    def predict_2step(self, temp_subspace_file, projected_file):
        op_labels_file = os.path.join(self.model_dir, 'op_labels.xml')
        bin_dir = config.bin_dir()
        run_cmd([os.path.join(bin_dir, 'aff_face_subspace_predict_2step'),
                 projected_file,
                 temp_subspace_file,
                 op_labels_file,
                 str(self.nmodes2),
                 str(self.topn),
                 str(self.thresh)])
        os.unlink(temp_subspace_file)
        return op_labels_file

    def subspace_predict(self, temp_subspace_file, projected_file):
        op_labels_file = os.path.join(self.model_dir, 'op_labels.xml')
        bin_dir = config.bin_dir()
        run_cmd([os.path.join(bin_dir, 'aff_face_subspace_predict'),
                 projected_file,
                 temp_subspace_file,
                 op_labels_file,
                 str(self.nmodes2),
                 str(self.topn),
                 str(self.thresh)])
        os.unlink(temp_subspace_file)
        return op_labels_file

    def train(self, features, labels):
        '''
            features = all the features for the faces, each row is a feature
            labels = a list of labels representing the id
                for each corresponding row in the features array
        '''
        if not self._is_trained:
            assert len(labels) == features.shape[0]
            self.model_dir = mkdtemp()
            # create feature file with given features
            # feature file needs to be such that the first dim in the label and
            # then each row in a feature
            self.feature_file = os.path.join(self.model_dir, 'features.xml')
            la = np.asarray([labels])
            cv_feats = np.transpose(np.concatenate((la, features.transpose())))
            mat_to_save = cv_feats.transpose().copy()
            # nFaces = mat_to_save->cols;
            # nDesc  = mat_to_save->rows-1;
            cv.Save(self.feature_file, cv.fromarray(mat_to_save))
            self.subspace_file = os.path.join(
                self.model_dir, 'learned_subspace.xml')
            self.face_subspace()
            self._is_trained = True

    def read_cv_xml(self, labels_file):
        return np.asarray(cv.Load(labels_file)).ravel()

    def test(self, features):
        subspace_file, projected_file = self.subspace_project(features)
        predicted_labels_file = self.predict_2step(
            subspace_file, projected_file)
        res = self.read_cv_xml(predicted_labels_file)
        os.unlink(predicted_labels_file)
        return res


class KNNScikit(AbstractClassifier):

    def __init__(self, neighbors=3, metric='euclidean', weights='distance',
                 **distance_kwargs):
        super(KNNScikit, self).__init__()
        self.knc = KNeighborsClassifier(
            n_neighbors=neighbors, metric=metric, weights=weights,
            **distance_kwargs)

    def train(self, features, labels):
        if not self._is_trained:
            self.labels_map = sorted(list(set(labels)))
            self.knc.fit(features, labels)
            self._is_trained = True

    def test(self, features):
        return self.knc.predict(features)

    def test_probability(self, features, min_prob_diff):
        y_pred = []
        prob_dist = self.knc.predict_proba(features)
        for row in prob_dist:
            idx = row.argsort()
            if row[idx[-1]] - row[idx[-2]] >= min_prob_diff:
                best_guess = self.labels_map[idx[-1]]
                y_pred.append(best_guess)
            else:
                y_pred.append(-1)
        return y_pred

    def get_neighbors(self, features, neighbors, dist=True):
        return self.knc.kneighbors(features, n_neighbors=neighbors,
                                   return_distance=dist)

    @classmethod
    def score_proba(cls, train_bag, test_bag, neg_bag, min_prob_diff=0.4,
                    **kwargs):
        knn = cls(**kwargs)
        knn.train(train_bag.feats, train_bag.labs)
        y_test_pred = knn.test_probability(test_bag.feats, min_prob_diff)
        test_score = accuracy_score(test_bag.labs, y_test_pred)
        y_neg_pred = knn.test_probability(neg_bag.feats, min_prob_diff)
        neg_score = 1 - accuracy_score(neg_bag.labs, y_neg_pred)
        return test_score, neg_score

    @classmethod
    def score(cls, train_bag, test_bag, **kwargs):
        knn = cls(**kwargs)
        knn.train(train_bag.feats, train_bag.labs)
        score = knn.knc.score(test_bag.feats, test_bag.labs)
        return score


class LibsvmClassifier(AbstractClassifier):

    """ Classifier wrapping around the Libsvm binaries
        For more information on usage of Libsvm binaries, please run
        LibsvmClassifier.usage()
        Note that all params supported by Libsvm can be passed
        as kwargs to the constructor
        Example:
            # This example creates a SVM classifier
            # where svm_type is epsilon-SVR
            # and returns probability_estimates
            clf = LibsvmClassifier(probability=1)
            data = np.random.rand(500, 5)
            labels = np.asarray([int(i > 0.5) for i in data[:, 0]])
            clf.train(data, labels)
            res = clf.test(data[:3, :])
            # res = array([1., 0., 1.])
            res = clf.test(data[:3, :], probability=True)
            # res = [(1.0, {0: 1.834e-05, 1: 0.9999}),
                    (0.0, {0: 0.7, 1: 0.3}),
                    (1.0, {0: 0.22, 1: 0.78})]
            # Res is a list of tuples (label, probabilty_dict)
    """

    def __init__(self, C=1, cache_size=100, coef0=0, degree=3, eps=0.001,
                 gamma=0, kernel_type=2, nr_weight=0, nu=0.5, p=0.1,
                 probability=0, shrinking=1, svm_type=0,
                 weight=None, weight_label=None):
        super(LibsvmClassifier, self).__init__()
        weight = weight or []
        weight_label = weight_label or []
        param_dict = {k: v for k, v in locals().items() if k != 'self'}
        self.set_svm_params(param_dict)
        self.model = None

    def usage(self):
        print u"""\nsvm_type : set type of SVM (default 0)\
            0 -- C-SVC\n1 -- nu-SVC2 -- one-class SVM\n3 -- epsilon-SVR\n4 --\
            nu-SVR\
            kernel_type : set type of kernel function (default 2)\n0 --\
            linear: u'*v\
            1 -- polynomial: (gamma*u'*v + coef0)^degree\n2 -- radial basis\
            function: exp(-gamma*|u-v|^2)\
            3 -- sigmoid : tanh(gamma*u'*v + coef0)\
            4 -- Precomputed\n5 -- intersection\
            degree : set degree in kernel function (default 3)\
            gamma : set gamma in kernel function (default 1/num_features)\
            coef0 : set coef0 in kernel function (default 0)\
            C : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR\
            (default 1)\
            nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR\
            (default 0.5)\
            p : set the epsilon in loss function of epsilon-SVR (default 0.1)\
            cache_size : set cache memory size in MB (default 100)\
            eps : set tolerance of termination criterion (default 0.001)\
            shrinking : whether to use the shrinking heuristics, 0 or 1\
            (default 1)\
            probability : whether to train a SVC or SVR model for probability\
            estimates, 0 or 1 (default 0)\
            weight : set the parameter C of class i to weight*C, for C-SVC\
            (default None)\
            weight_label : (for C_SVC) weight_label, nr_weight, and weight are\
            used to change the penalty\
            for some classes (If the weight for a class is not changed, it\
            is set to 1). This is useful for\
            training classifier using unbalanced input data or with asymmetric\
            misclassification cost.(default None )\
            nr_weight : (for C_SVC) the number of elements in the array\
            weight_label and weight. Each weight[i] corresponds\
            to weight_label[i], meaning that the penalty of class\
            weight_label[i] is scaled by a factor of weight[i] (default 0)"""

    def set_svm_params(self, param_dict):
        param_str = ' '.join(k for k in param_dict.keys())
        Params = namedtuple('Params', param_str)
        self.params = Params(**param_dict)
        self._svm_parameter = svm.svm_parameter(
            **{k: getattr(self.params, k) for k in self.params._fields})

    def __deepcopy__(self, memo):
        cls = self.__class__
        clf = cls()
        clf.set_svm_params(self.params._asdict())
        if self.model is not None:
            try:
                h, model_path = mkstemp()
                os.close(h)
                self.model.save(model_path)
                clf._load_model_file(model_path)
            finally:
                os.unlink(model_path)
        return clf

    def train(self, features, labels):
        assert isinstance(labels, np.ndarray), "labels should be numpy array"
        features = self._cleanse_features(features)
        problem = svm.svm_problem(labels.tolist(), features)
        self.model = svm.svm_model(problem, self._svm_parameter)

    def test(self, features, probability=False):
        assert self.model is not None, 'Model not set, call training first'
        if probability:
            return self.predict_probability(features)
        else:
            return self.predict(features)

    def _cleanse_features(self, features):
        """ Features should be eithter a numpy array where
            each row represents a feature or it should be
            a list of lists
        """
        if isinstance(features, np.ndarray):
            if len(features.shape) == 1:
                features = features.reshape(1, len(features))
            assert len(features.shape) == 2, features.shape
            features = features.tolist()
        elif isinstance(features, list):
            assert isinstance(features[0], dict)
        else:
            raise ValueError(
                "Features can be a numpy array or a list of dicts")
        return features

    def predict_probability(self, features):
        assert self.model is not None, "Model is not trained"
        features = self._cleanse_features(features)
        return [self.model.predict_probability(f) for f in features]

    def predict(self, features):
        assert self.model is not None, "Model is not trained"
        features = self._cleanse_features(features)
        return np.asarray([self.model.predict(f) for f in features])

    def save_to_file(self, file_name):
        self.model.save(file_name)

    def _load_model_file(self, model_file_path):
        assert os.path.exists(model_file_path), model_file_path
        self.model = svm.svm_model(model_file_path)

    @classmethod
    def load_from_file(cls, file_path):
        # Note that loading a model file directly will not set the svm_params
        # correctly.
        assert os.path.exists(file_path)
        clf = cls()
        clf.params = None
        clf._load_model_file(file_path)
        return clf

    @classmethod
    def parse_support_vectors(cls, model_file):
        labels = []
        support_vecs = []
        with open(model_file, 'r') as f:
            start_flag = 0
            for line in f:
                vec = []
                if line == 'SV\n':
                    start_flag = 1
                    continue
                if start_flag:
                    elems = line.split()
                    for e in elems[1:]:
                        points = e.split(":")
                        vec.append(float(points[1]))
                    label = float(elems[0])
                    labels.append(label)
                    support_vecs.append(vec)
        return labels, np.asarray(support_vecs)


class RandomForest(AbstractClassifier):
    """ Classifier wrapper around the sklearn RandomForestClassifier """

    def __init__(self, n_estimators=10, criterion='gini',
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features='auto', bootstrap=True, oob_score=False,
                 n_jobs=1, random_state=None, verbose=0, min_density=None,
                 compute_importances=None):
        """ this classifier is a wrapper on the sklearn RandomForestClassifier,
        parameters are the directly those offered by that function """

        super(RandomForest, self).__init__()
        self.rfc = RandomForestClassifier(
            n_estimators, criterion, max_depth, min_samples_split,
            min_samples_leaf, max_features, bootstrap, oob_score, n_jobs,
            random_state, verbose, min_density, compute_importances)

    def train(self, features, labels):
        """
        Train the random forest classifier with the given training data
        Args:
            features: each row is the feature of one sample
            labels: nparray with one label per sample
        Asserts: if the input params are not numpy arrays
        """
        assert isinstance(labels, np.ndarray), "labels should be numpy array"
        assert isinstance(features, np.ndarray),\
            "features should be numpy array"
        self.rfc.fit(features, labels)

    def test(self, features, probability=False):
        """
        Run predict for input features.
        Args:
            features: each row is the feature of one sample
            probability: bool,
                if true returns probability of each possible class,
                otherwise it only returns the label with highest probability
        Returns:
            predicted label or list of probability for each possible label
        Asserts: if the model has not been loaded/trained
        """
        assert self.rfc.estimators_, 'Model not set, call training first'
        if probability:
            return self.rfc.predict_proba(features)
        else:
            return self.rfc.predict(features)


class DecisionStump(AbstractClassifier):
    """ Decision Stump Classifier """

    def __init__(self, threshold=0):
        self.threshold = threshold

    def train(self, features, labels):
        """ Compute parameters for scaling test data
        Args:
            features: a matrix of features rows
            labels: a list of int. Labels start from class 1...n
            as label 0 is considered as No class
        """
        msg = "An array of features is needed for classification"
        assert isinstance(features, np.ndarray), msg
        self.num_labels = features.shape[1] + 1

    def test(self, features):
        """
            It returns the position from 1 to N (dim) with highest value.
            If the highest value is lower than threshold, returns 0

            Args:
                features: a numpy array. Each row is a feature vector
            Returns:
                pred_labels: list with index of highest label (from 1 to N)
                    or 0 if value < threshold for each input feature
            Assertions:
                AssertionError if features is not an ndarray and if the
                dimensions are not consistent

        """
        assert isinstance(features, np.ndarray), "features is a numpy array"
        msg = "wrong feature dimensions"

        if features.ndim == 1:
            features = np.array(features, ndmin=2)
        assert features.shape[1] == self.num_labels - 1, msg

        pred_labels = [np.argmax(p) if np.max(p) >= self.threshold
                       else -1 for p in features]
        return pred_labels
