"""
Class to train and test classifier that uses cnn features + SVM/NN/threshold
"""
import os
import numpy as np
import shutil
from logging import getLogger

from affine.detection.model import std_cross_validation
from affine.detection.model import LibsvmClassifier, RandomForest,\
    DecisionStump
from affine.detection.model.mlflow.flow\
    import Step, Flow, FutureLambda, FutureFlowInput
from affine.detection.vision.utils.scene_functions import get_config
from affine.detection.model_worker.tasks.data_processor_task import\
    DataProcessorClient, convert_files_to_bin, Pool
from affine.model.classifier_models import CaffeModel
from affine.detection.model_worker.utils import batch_list, merge_list

__all__ = ['CnnClassifierFlowFactory']

CNN_TIMEOUT = 120

logger = getLogger(__name__)

CNNCLF_CFG_SPEC = """
    [train_params]
        training_folders = string_list(min=2,\
            default=list('negative', 'positive'))
        target_label_ids = int_list(default=list(-1, -1))
    [cnn_model_params]
        model_name = string(min=0, max=256)
        layer_name = string(min=0, max=256, default='')
        oversampling = boolean(default=False)
    [classifier_params]
        classifier_name = string(min=2, default='svm')
        clf_type = integer(default=2)
        svm_kernel = integer(default=2)
        svm_c = integer(default=1)
        svm_gamma = float(default=0)
        rf_n_estimators = integer(default=10)
        rf_criterion = string(min=2, default='gini')
        rf_min_split = integer(default=2)
        rf_min_leaf = integer(default=1)
        ds_threshold = float(default=0)
    [crossval_params]
        test_size = float(default=0.1)
        valid_th = float(min=0.0, default=0.7)
    [acceptance_params]
        max_num_frames = integer(min=0, default=0)
        accept_th = float(min=0.0, default=0.5)
        min_accept = integer(min=1, default=1)
    """


def run_remove_nans(desc, labels=None):
    """
    Remove rows in both desc and labels, if they contain Nans in desc
    Args:
        desc: 2-d np array with descriptors (one row per data point)
        labels: 1-d array, one label per descriptor/data point
    Returns:
        data (dict), with the format data = {'X': new-desc-2Darray,
        'Y': new-label-array, 'Nan_idx': list of image indexes removed}

    Asserts:
        when labels is not empty, it will assert if the number of descriptors
        is not the same as number of labels
    """
    if labels is None:
        labels = []
    X = np.array(desc)
    if len(labels) == 0:
        labels = [-1] * X.shape[0]
    Y = np.array(labels)
    assert (X.shape[0] == Y.shape[0]), \
        "Number of descriptors (%d) and labels (%d) should be the same" %\
        (X.shape[0], Y.shape[0])

    nan_row_count = np.sum(np.isnan(X), axis=1)
    nan_image_idx = [
        i for i in range(X.shape[0]) if nan_row_count[i] > 0]
    if nan_image_idx:
        nan_image_idx.sort(reverse=True)
        for bad_indx in nan_image_idx:
            X = np.delete(X, bad_indx, 0)
            Y = np.delete(Y, bad_indx, 0)
    data = {'X': X, 'Y': Y, 'Nan_idx': nan_image_idx}
    return data


def add_result_for_nans(predicted_labels, indx_nan, num_results):
    """ Helper function to handle Nans that may show up in feature vectors.
        If that happens, we can't run "predict" on those features, but for
        consistency, this function adds a -1 as the prediction for features
        that contained Nans.

        Args:
            predicted_labels: list of int, predicted labels for not Nan features
            indx_nan: indexes from original set of features which contained Nans
            num_results: number of expected predictions
                (i.e., original number of input features)
        Returns:
            expanded_labels: list of int representing the predicted labels,
                including -1 for each indx_nan in the expected position
                within the predicted_labels list
    """
    expanded_labels = predicted_labels
    if len(indx_nan) > 0:
        logger.info('Number of Nans found: %d' % len(indx_nan))
        indx_ok = set(range(num_results)) - set(indx_nan)
        expanded_labels = [-1] * num_results
        for i, j in zip(indx_ok, predicted_labels):
            expanded_labels[i] = j

    return expanded_labels


class CnnClassifierFlowFactory(object):

    """
    Flow to run a classifier that uses generic holistic features and an SVM.
    It enables the use of features computed from a gpu caffe feature server.
    """

    CONFIG_FILE = 'Configfile.cfg'
    CLASSIFIER_FILE = {0: 'svm.pickle', 1: 'rf.pickle', 2: 'ds.pickle'}
    CLASSIFIERS = {0: LibsvmClassifier, 1: RandomForest, 2: DecisionStump}

    def __init__(self, configfile_name):
        """
        Load and initialize all the configuration params for feature extraction
        and classifiers. Default classifier is a multiclass rbf-svm.

        Args:
            configfile_name: string with full-path name of the config file
        Returns:
        Raises/Assertions:
        """
        assert os.path.exists(configfile_name), \
            'Config file %s does not exist' % (configfile_name)

        self._load_config_file(configfile_name)
        self.model_dir = None
        self.fullpath_input_configfile = configfile_name

    def _load_classifiers_params(self, cfg_obj):
        """
        Load the correct classifier and parameters, given the config_file
        Args:
            cfg_obj: Config obj
        """
        self.clf_type = cfg_obj['classifier_params']['clf_type']
        self.prob_threshold = cfg_obj['classifier_params']['ds_threshold']

        if self.clf_type == 0:
            kernel_type = cfg_obj['classifier_params']['svm_kernel']
            gamma = cfg_obj['classifier_params']['svm_gamma']
            c = cfg_obj['classifier_params']['svm_c']
            self.clf = LibsvmClassifier(C=c, kernel_type=kernel_type,
                                        gamma=gamma, probability=1)
        elif self.clf_type == 1:
            estimators = cfg_obj['classifier_params']['rf_n_estimators']
            criterion = cfg_obj['classifier_params']['rf_criterion']
            min_split = cfg_obj['classifier_params']['rf_min_split']
            min_leaf = cfg_obj['classifier_params']['rf_min_leaf']
            self.clf = RandomForest(n_estimators=estimators,
                                    criterion=criterion,
                                    min_samples_split=min_split,
                                    min_samples_leaf=min_leaf)
        else:
            self.clf = DecisionStump(threshold=self.prob_threshold)

    def _load_config_file(self, configfile_name):
        """
        Fill the classifier params from the given config file

        Args:
            configfile_name: string with full-path name of the config file
        Raises/Assertions:
            AssertionError: get_config raises AssertionError if configfile \
                has bad formatting: 'Config file validation failed'
        """
        cfg_obj = get_config(configfile_name, spec=CNNCLF_CFG_SPEC.split('\n'))
        if cfg_obj:
            target_label_info = cfg_obj['train_params']['target_label_ids']
            assert len(target_label_info) % 2 == 0, \
                "target_label_ids should have an even number of elements, "\
                "representing pairs of classifier output and DB label id, "\
                " e.g., 0, 1004, 1, 1006"
            self.target_labels = dict(zip(*[iter(target_label_info)] * 2))
            if len(self.target_labels) > 3:
                label_list = str(self.target_labels.values()[0:3]) + '...'
            else:
                label_list = str(self.target_labels.values())
            logger.info("Loaded classifier has %d target labels (%s)" %
                        (len(self.target_labels), label_list))

            self.cnn_model_name = cfg_obj['cnn_model_params']['model_name']
            self.cnn_layer_name = cfg_obj['cnn_model_params']['layer_name']
            self.oversampling = cfg_obj['cnn_model_params']['oversampling']
            cm = CaffeModel.by_name(self.cnn_model_name)
            assert cm,\
                "%s model, from Config file, does not exist in the DB" % \
                (self.cnn_model_name)
            logger.info('Using %s' % (self.cnn_model_name))
            self._load_classifiers_params(cfg_obj)
            self.crossval_tsize = cfg_obj['crossval_params']['test_size']
            self.crossval_accept = cfg_obj['crossval_params']['valid_th']

            self.max_frames_per_video = \
                cfg_obj['acceptance_params']['max_num_frames']
            self.accept_th = cfg_obj['acceptance_params']['accept_th']
            self.min_accept = cfg_obj['acceptance_params']['min_accept']

    def save_model(self, output_dir):
        """
        Save currently loaded classifier model.

        This function stores a pickle files for the SVM objects (svm.pickle)
        and a copy of the config file used for the current classifier as:
            Configfile.cfg

        Args:
            output_dir: string, folder where we want to save the model files.
                It is expected to be an EMPTY FOLDER
                If it does not to exist, it will be created.
        """
        if self.clf:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            logger.info('Saving config %s in %s' %
                        (self.fullpath_input_configfile, output_dir))

            dst_config_file = os.path.join(output_dir, self.CONFIG_FILE)

            if self.fullpath_input_configfile is not dst_config_file:
                shutil.copy(self.fullpath_input_configfile, dst_config_file)
            self.clf.save_to_file(
                os.path.join(output_dir, self.CLASSIFIER_FILE[self.clf_type]))
            self.model_dir = output_dir

    def load_model(self, model_dir):
        """
        Load a previously trained classifier model.

        Args:
            model_dir: folder with all required files to load the model.
        Raises/Assertions:
            Asserts if any model required file (config or model) does not exist
            Asserts if the config_file fails the formatting check.
        """

        assert os.path.exists(model_dir), \
            "Folder %s with model files does not exist" % (model_dir)

        config_file = os.path.join(model_dir, self.CONFIG_FILE)
        model_filename = os.path.join(model_dir,
                                      self.CLASSIFIER_FILE[self.clf_type])
        assert os.path.exists(config_file) and os.path.exists(model_filename),\
            "Required model files (%s, %s) not found in model folder %s" %\
            (config_file, model_filename, model_dir)

        self._load_config_file(config_file)
        self.clf = self.CLASSIFIERS[self.clf_type].load_from_file(
            model_filename)
        self.model_dir = model_dir
        self.fullpath_input_configfile = config_file

    def _get_features_using_pool(self, input_img_data, action, feat_layer=None,
                                 async=True, timeout=CNN_TIMEOUT, batch_size=10):
        dpc = DataProcessorClient(self.cnn_model_name)
        pool = Pool(dp_client=dpc, num_calls=3)  # NUM_ASYNC_CALLS default = 10??
        if action == 'extract':
            kwargs = {'feat_layer': feat_layer, 'async': async, 'timeout': timeout}
        else:
            kwargs = {'async': async, 'timeout': timeout}

        list_data_batches = batch_list(input_img_data, batch_size=batch_size)
        params = []
        for data_batch in list_data_batches:
            params += [[(data_batch,), kwargs]]
        features_list = pool.map_async(func_name=action, arg_list=params)

        features = merge_list(features_list)

        return features

    def get_compute_descriptors_flow(self):
        """
        Flow to compute descriptors for all videos in the input list which will
        be provided in the required FutureFlowInput "image_paths"

        Returns:
            flow object
        """
        f = Flow("Compute features flow")
        convert_to_bin = Step("convert_to_bin", convert_files_to_bin)
        f.add_step(convert_to_bin)

        cnn_features = Step(
            'features', self._get_features_using_pool, None)
        f.add_step(cnn_features)
        if len(self.cnn_layer_name):
            logger.info('Extracting cnn %s features' % (self.cnn_layer_name))
            f.connect(convert_to_bin, cnn_features, convert_to_bin.output,
                      action='extract', feat_layer=self.cnn_layer_name)
        else:
            logger.info('Predicting cnn (last deploy layer)')
            f.connect(convert_to_bin, cnn_features, convert_to_bin.output,
                      action='predict')

        f.start_with(convert_to_bin, FutureFlowInput(f, 'image_paths'),
                     resize=(256, 256))
        f.output = cnn_features.output
        return f

    def get_train_flow(self):
        """
        Get flow to train this classifier.

        Training images and labels will be provided in the required
            FutureFlowInput "image_paths" and "image_labels".
        Trained model is stored in required FutureFlowInput "output_model_dir"

        Returns:
            flow object
        """

        f = Flow('Cnn based Training Flow')
        desc_flow = self.get_compute_descriptors_flow()
        features = Step('get_features', desc_flow, 'run_flow')
        remove_nans = Step('remove_nans', run_remove_nans, None)
        crossval = Step('crossval', std_cross_validation, None)
        final_clf = Step('train_final_clf', self.clf, 'train')
        save_m = Step('save_model', self, 'save_model')

        for step in [features, crossval, final_clf, save_m, remove_nans]:
            f.add_step(step)

        f.start_with(features, image_paths=FutureFlowInput(f, 'image_paths'))
        f.connect(features, remove_nans,
                  desc=features.output,
                  labels=FutureFlowInput(f, 'image_labels'))
        np_feat = FutureLambda(remove_nans.output, lambda x: np.array(x['X']))
        np_labels = FutureLambda(
            remove_nans.output, lambda x: np.array(x['Y']))

        f.connect(remove_nans, crossval, self.clf, np_feat, np_labels)
        f.connect(crossval, final_clf, np_feat, np_labels)
        f.connect(final_clf, save_m, FutureFlowInput(f, 'output_model_dir'))

        f.output = crossval.output
        return f

    def _test_rf_prob(self, features):
        """
        Helper function to check if it's needed to reject the predicton
        of the random forest if the best label probability is < acceptance th
        Note that it returns -1 if no label matches this,
        assuming that actual label indexes positive numbers
        """
        prob_labels = self.clf.test(features, probability=True)
        pred_labels = [np.argmax(p) if np.max(p) >= self.prob_threshold
                       else -1 for p in prob_labels]
        return pred_labels

    def get_process_video_flow(self):
        """
        Get flow to run tests with this classifier.

        The flow predicts the label for each image in the input list
        (provided in the required FutureFlowInput "image_paths")
        This method ASSUMES that the classifier is already trained or loaded,
        otherwise the process video flow cant be created.

        Returns:
            flow object
        Raises/Assertions:
            asserts if classifier model does not exist (e.g., is not loaded)
        """
        assert self.model_dir, \
            "Classifier is not trained or loaded, prediction can't be done. \
            You need to run a step that loads the model before this."

        f = Flow('Cnn based Testing Flow')
        desc_flow = self.get_compute_descriptors_flow()
        features = Step('get_features', desc_flow, 'run_flow')
        remove_nans = Step('remove_nans', run_remove_nans, None)
        if self.clf_type == 1:
            predict = Step('test', self._test_rf_prob, None)
        else:
            predict = Step('test', self.clf, 'test')
        add_nan_result = Step('add_result_for_nan', add_result_for_nans, None)

        for step in [features, remove_nans, predict, add_nan_result]:
            f.add_step(step)

        f.start_with(features, image_paths=FutureFlowInput(f, 'image_paths'))
        num_results = FutureLambda(features.output, lambda x: np.shape(x)[0])

        f.connect(features, remove_nans, desc=features.output)
        np_feat = FutureLambda(remove_nans.output, lambda x: np.array(x['X']))
        indx_nan = FutureLambda(
            remove_nans.output, lambda x: np.array(x['Nan_idx']))

        f.connect(remove_nans, predict, np_feat)
        f.connect(predict, add_nan_result, num_results=num_results,
                  predicted_labels=predict.output, indx_nan=indx_nan)

        f.output = add_nan_result.output

        return f
