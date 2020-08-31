"""
Class to train and test classifier for motion-color based video detection
"""
import os
import numpy as np
import shutil

from logging import getLogger
from scipy import stats
from affine.detection.model.classifiers import LibsvmClassifier
from affine.detection.model.features import ColorHistFeatureExtractor
from affine.detection.model.features import OpticalFlowFeatureExtractor
from affine.detection.vision.video_motioncolor.video_sampler import \
        VideoFFmpegSampler
from affine.detection.vision.utils.scene_functions import get_config

__all__ = ['VideoInfoObject', 'VideoMotionColorClassifier']

logger = getLogger(__name__)

VIDEOG_CFG_SPEC = """
    [sampling_params]
        duration = integer(min=1, max=90, default=30)
        fps = float(min=0.1, max=60, default=2)
        frame_scale = float(min=0.0001, max=4096, default=320)
    [opt_flow_params]
        get_masked_flow = boolean(default=True)
        flow_numBins = integer(min=8, max=32, default=8)
    [color_hist_params]
        color_numBins = integer(min=8, max=256, default=16)
        colorspaces = string_list((min=1, max=3, default=list('RGB', 'Lab'))
        num_frames_per_video = integer(min=1, max=30, default=5)
    [svm_params]
        svm_type = integer(default=0)
        svm_kernel = integer(default=2)
        svm_gamma = float(default=0.7)
    [crossval_params]
        test_size = float(default=0.1)
        valid_th = float(min=0.0, default=0.7)
    [classif_params]
        conf_th = float(default=0.65)
        accept_th = float(min=0.0, default=0.5)
        ratio_v = float(min=0.1)
    """


class VideoInfoObject(object):

    """
    Auxiliary class to organize the necesary video info \
    that the VideoMotionColorClassifier needs
    """

    def __init__(self, video_id, videopath, framespath, length):
        self.video_id = video_id
        self.video_name_path = videopath
        self.frames_folder_path = framespath
        self.video_length = length


class VideoMotionColorClassifier(object):

    """
    Classifier to detect if a video belongs to a certain class 
    that has representative motion and color patterns. For example, if it
    is (contains) a videogame play screen capture.
    NOTE: the input for train and test are lists of the video objects defined \
    in the class VideoInfoObject, NOT the Video class in affine.detection.model
    """

    CONFIG_FILE = 'Configfile.cfg'
    COLOR_FILE = 'Color_svm.pickle'
    MOTION_FILE = 'Motion_svm.pickle'

    def __init__(self, configfile_name):
        """
        Load and initialize all the configuration params 
        (for sampling, feature extraction and classifiers)

        Args:
            configfile_name: string with full-path name of the config file, 
                default value is './Configfile'

        Returns:

        Raises/Assertions:
        """
        assert os.path.exists(configfile_name), \
            'Config file %s does not exist' % (configfile_name)

        self.load_config_file(configfile_name)

        # initialize descriptor steps
        self.color_extractor = ColorHistFeatureExtractor(
            color_spaces_used=self.colorspaces, numBins=self.color_numBins)
        self.flow_extractor = OpticalFlowFeatureExtractor(
            get_masked_flow=self.get_masked_flow, numBins=self.flow_numBins)

        self.desc_dim_c = self.color_extractor.get_desc_length() \
            * self.num_frames_per_video
        self.desc_dim_m = self.flow_extractor.get_desc_length()

        # initialize classifier steps
        self.clf_c = LibsvmClassifier(svm_type=self.svm_type,
                                      kernel_type=self.kernel_type,
                                      gamma=self.gamma,
                                      probability=1)

        self.clf_m = LibsvmClassifier(svm_type=self.svm_type,
                                      kernel_type=self.kernel_type,
                                      gamma=self.gamma,
                                      probability=1)

        self.model_dir = None
        self.prec = -1

        self.fullpath_input_configfile = configfile_name

    def load_config_file(self, configfile_name):
        """
        Fill the classifier params from the given config file

        Args:
            filename: string with full-path name of the config file
        Returns:
            True on successful config load

        Raises/Assertions:
            AssertionError: get_config raises AssertionError if configfile \
                has bad formatting: 'Config file validation failed'
        """
        cfg_obj = get_config(configfile_name, spec=VIDEOG_CFG_SPEC.split('\n'))

        self.sampling_duration = cfg_obj['sampling_params']['duration']
        self.sampling_fps = cfg_obj['sampling_params']['fps']
        self.sampling_scale = cfg_obj['sampling_params']['frame_scale']

        self.get_masked_flow = cfg_obj[
            'opt_flow_params']['get_masked_flow']
        self.flow_numBins = cfg_obj['opt_flow_params']['flow_numBins']

        self.color_numBins = cfg_obj['color_hist_params']['color_numBins']
        self.colorspaces = cfg_obj['color_hist_params']['colorspaces']
        self.num_frames_per_video = cfg_obj[
            'color_hist_params']['num_frames_per_video']

        self.svm_type = cfg_obj['svm_params']['svm_type']
        self.kernel_type = cfg_obj['svm_params']['svm_kernel']
        self.gamma = cfg_obj['svm_params']['svm_gamma']

        self.crossval_test_size = cfg_obj['crossval_params']['test_size']
        self.crossval_th = cfg_obj['crossval_params']['valid_th']

        self.confidence_th = cfg_obj['classif_params']['conf_th']
        self.ratio_motioncolor_votes = cfg_obj['classif_params']['ratio_v']
        self.accept_th = cfg_obj['classif_params']['accept_th']

    def train(self, X, Y):
        """
        Train classifier with input descriptors and labels

        This function modifies the object color (self.clf_c)
        and motion (self.clf_m) SVM objects

        Args:
            X: 2-dimensional np.array with data descriptors. 
                Each row contains the descriptor of one of the training samples.
            Y: 1-d np.array with the labels of the training data. 
                Each label corresponds to the same index row in the matrix X.
                Note that this function assumes that X and Y have the same number of elements
                len(Y) == X.shape[0] == number of training samples
        Returns:

        Raises/Assertions:
        """

        X_Motion, X_Color = self._splitX(X)

        self.clf_m.train(X_Motion, Y)

        Y_color = []
        for l in Y:
            Y_color = np.append(
                Y_color, np.ones((1, self.num_frames_per_video)) * l)
        self.clf_c.train(X_Color, Y_color)

    def save_model(self, output_dir):
        """
        Save current trained classifier models.

        This function stores two pickle files, one for each of the internal SVMs 
        color (self.clf_c) and motion (self.clf_m) SVM objects:
            Color_svm.pickle and Motion_svm.pickle
        and a copy of the config file used for the current classifier as:   
            Configfile.cfg

        Args:
            output_dir: folder where we want to save the model files.
                It is expected to be an EMPTY FOLDER 
                or to not exist (in this case it will be created)

        Returns:

        Raises/Assertions:
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info('Saving config %s' % (self.fullpath_input_configfile))
        dst_config_file = os.path.join(output_dir, self.CONFIG_FILE)
        if self.fullpath_input_configfile != dst_config_file:
            shutil.copy(self.fullpath_input_configfile, dst_config_file)
        self.clf_c.save_to_file(os.path.join(output_dir, self.COLOR_FILE))
        self.clf_m.save_to_file(os.path.join(output_dir, self.MOTION_FILE))
        self.model_dir = output_dir

    def load_model(self, model_dir):
        """
        Load a previously trained classifier model.

        This function loads the config file with general configuration of the
        classifier and loads the two pickle files, one for each of the internal 
        SVM objects(color (self.clf_c) and motion (self.clf_m))

        Args:
            config_file: string, full-path name to configuration filename. Default is
                CFG_FILE = "Configfile.cfg"
            svm_file: string, SUFFIX of the svm files. Default is SVM_FILE = "Modelfile.pickle"
                Color model file will be a string appending 'color_' in front of this string
                Motion model file will be a string appending 'motion_' in front of this string

        Returns:

        Raises/Assertions:
            Asserts if config_file, or any of the two svm files do not exist.
            Asserts if the config_file fails the formatting check.
        """

        assert os.path.exists(model_dir), \
            "Folder %s with model files does not exist" % (model_dir)

        config_file = os.path.join(model_dir, self.CONFIG_FILE)
        assert os.path.exists(config_file), \
            "Config file not found in model folder %s" % (model_dir)

        colormodel_filename = os.path.join(model_dir, self.COLOR_FILE)
        assert os.path.exists(colormodel_filename), \
            "COLOR svm file not found in model folder %s" % (model_dir)

        motionmodel_filename = os.path.join(model_dir, self.MOTION_FILE)
        assert os.path.exists(motionmodel_filename), \
            "MOTION svm file not found in model folder %s" % (model_dir)

        self.load_config_file(config_file)

        self.clf_c = LibsvmClassifier.load_from_file(colormodel_filename)
        self.clf_m = LibsvmClassifier.load_from_file(motionmodel_filename)

        self.model_dir = model_dir
        self.fullpath_input_configfile = config_file

    def classify_videos(self, list_videoObj):
        """
        Predict the label for each input VideoInfoObject object.
        This ASSUMES that the classifier is already trained or loaded, otherwise it will assert

        Args:
            list_videoObj: list of VideoInfoObject objects with the info of the test videos

        Returns:
            list_of_labels: list of int, 
                predicted label for each of the test videos in list_videoObj
                label 0 is the target label (e.g., videogame), label 1 other types

        Raises/Assertions:
            asserts if the classifier model does not exist (e.g., is not loaded)

        """
        assert self.model_dir != None, \
            "Classifier is not trained or loaded, prediction can't be done"

        num_videos = len(list_videoObj)
        y_pred_M_C = [1] * num_videos

        logger.info(' 1 - compute TEST descriptors for %d tests' %
                    (num_videos))
        X_test, correct_video_indx = self.compute_descriptors(list_videoObj)

        num_videos_ok = len(correct_video_indx)
        logger.info(' 2 - predict TEST label for %d tests' % (num_videos_ok))
        if type(X_test) == np.ndarray and X_test.shape[0] > 0:
            y_pred_M_C_correct_indx = self.test(X_test)

        for i1, i2 in zip(correct_video_indx, range(num_videos_ok)):
            y_pred_M_C[i1] = y_pred_M_C_correct_indx[i2]

        return y_pred_M_C

    def compute_descriptors(self, videoObj_list):
        """
        Computes the descriptors for all videos in the input list.

        This function creates a ColorHistFeatureExtractor and a OpticalFlowFeatureExtractor. 
        Then, for each input video, it obtains the necessary frames 
        (a few sparse frames for the Color Feature, and more and more frequent frames to obtain the Optical Flow feature)
        and extract the features on them concatenating all the descriptors corresponding to the same video 
        into a single row of the result matrix.

        This function also assigns the value to self.desc_dim_c and self.desc_dim_m,
        to describe the length of each of the individual descriptors used

        Args:
            videoObj_list: list of VideoInfoObject objects with the basic info of the input videos
        Returns:
            X: 2-dimensional np.array with data descriptors. 
                Each row contains the computed descriptor of each input video,
                that was valid (e.g., if too short, descriptors cant be computed)
                If all videos are correct, it will have as many rows as videos in the list
            correct_indx: list of the position of valid videos in the input videoObj_list
                It indicates to video corresponds to each of the descriptor returned
        Raises/Assertions:
        """

        desc_length = self.desc_dim_c + self.desc_dim_m

        X = np.zeros((1, desc_length))
        correct_video_indx = []

        for i, v in enumerate(videoObj_list):
            v_name = v.video_name_path
            v_length = v.video_length
            possible_offsets = np.array([v_length / 2, 60])
            off = np.min(possible_offsets[possible_offsets >= 0])
            logger.info('\t 1.1. Sampling video %s (offset %.2f)'
                        % (v.video_name_path, off))

            sampler = VideoFFmpegSampler(
                v_name, duration=self.sampling_duration, offset=off,
                fps=self.sampling_fps, scale=self.sampling_scale)
            list_of_frames = sampler.sample(output_dir=v.frames_folder_path)

            logger.info('\t 1.2. Computing descriptors')
            num_frames = len(list_of_frames)
            X_row = []
            if num_frames > self.num_frames_per_video:

                X_tmp_f = self.flow_extractor.extract(list_of_frames)
                # this can happen if we only have 1 frame
                if np.sum(X_tmp_f) == 0:
                    logger.error('No MOTION DESCRIPTORS computed for %s'
                                 % (v_name))
                else:
                    step = 1
                    if self.num_frames_per_video > 1:
                        step = num_frames / self.num_frames_per_video

                    list_imagefiles_color = \
                        list_of_frames[0:num_frames:step]
                    list_imagefiles_color = \
                        list_imagefiles_color[0:self.num_frames_per_video]

                    X_tmp_c = self.color_extractor.extract(
                        list_imagefiles_color)
                    X_tmp_c = X_tmp_c.reshape(
                        (1, X_tmp_c.shape[0] * X_tmp_c.shape[1]))

                    X_row = np.append(X_tmp_f, X_tmp_c, axis=1)

                if len(X_row) > 0 and X_row.shape[1] == desc_length:
                    correct_video_indx = correct_video_indx + [i]
                    X = np.append(X, X_row, axis=0)
                    # only appends the row if the descriptor computation was
                    # succesful
            else:
                logger.error('There are not enough frames for this video, \
                    No descriptor computed for %s' % (v_name))

        # delete tmp initial empty row
        X = np.delete(X, 0, axis=0)

        logger.info('End descriptor computation: X %s' % (str(np.shape(X))))
        return X, correct_video_indx

    def _splitX(self, X):
        """
        Split the given video descriptor matrix into Color and Motion descriptor matrices.
        Note: THIS ASSUMES that X contains a concatenated Motion and Color descriptor \
        of the corresponding size, i.e., each row should be like:
            [motion_descriptor color_descriptor]

        Args:
            X: 2-dimensional np.array with data descriptors. 
                Each row contains the descriptor of one of the input videos.

        Returns:
            X_Motion, X_Color: 2-dimensional np.arrays with data descriptors. 
                Each of them is a subset of X (same number of rows, but only
                the corresponding columns to the type of descriptor info they contain)

        Raises/Assertions:
            Asserts if dimensions of X do not match the class descriptor extraction options
            i.e., each row in X should be at least self.desc_dim_m + self.desc_dim_c

        """
        width = X.shape[1]
        desc_length = self.desc_dim_m + self.desc_dim_c

        assert (width == desc_length), \
        "Descriptor matrix does not contain a number of expected descriptors"

        dimM = self.desc_dim_m
        X_Motion = X[:, 0:dimM]
        X_Color = X[:, dimM:]

        # No need to reshape when there is one or less color frames
        if self.num_frames_per_video > 1:
            X_Color = X_Color.reshape(
                X_Color.shape[0] * self.num_frames_per_video,
                X_Color.shape[1] / self.num_frames_per_video)

        return X_Motion, X_Color

    def test(self, X):
        """
        Estimates the predicted label for each input descriptor.

        Args:
            X: 2-dimensional np.array with data descriptors. 
                Each row contains the descriptor of one of the test samples.

        Returns:
            list of predicted label (or labels) for each of the input descriptors

        Raises/Assertions:
        """
        assert type(X) == np.ndarray and X.shape[0] > 0, \
            'X has to be a numpy array and contain at least 1 row of data'

        numTestVideos = X.shape[0]
        X_Motion, X_Color = self._splitX(X)
        yProb_pred_M_tmp = self.clf_m.predict_probability(X_Motion)
        yProb_pred_C_tmp = self.clf_c.predict_probability(X_Color)
        yProb_pred_M = np.array([d.values() for _, d in yProb_pred_M_tmp])
        yProb_pred_C = np.array([d.values() for _, d in yProb_pred_C_tmp])

        y_pred_Motion = []
        y_pred_Color = []
        y_pred_M_C = []

        for v_index in range(0, numTestVideos):
            preds_motion = yProb_pred_M[v_index, :]
            l_M = np.argmax(preds_motion)
            y_pred_Motion = np.append(y_pred_Motion, l_M)

            num_frames = self.num_frames_per_video
            start_row = v_index * num_frames
            preds_current_frames = \
                np.array(yProb_pred_C[start_row:start_row + num_frames, :])
            avg_preds_color = np.sum(preds_current_frames, axis=0) / num_frames
            l_C = np.argmax(avg_preds_color)
            y_pred_Color = np.append(y_pred_Color, l_C)

            max_color_votes = num_frames

            if max_color_votes:
                max_motion_votes = int(
                    max_color_votes * self.ratio_motioncolor_votes)
            else:
                max_motion_votes = 1

            motion_votes = []
            if np.max(preds_motion) > self.confidence_th:
                motion_votes = [l_M] * max_motion_votes

            color_votes = []
            for fr in range(num_frames):
                conf = np.max(preds_current_frames[fr, :])
                l = np.argmax(preds_current_frames[fr, :])
                if conf > self.confidence_th and l == l_C:
                    color_votes = np.append(color_votes, l)

            total_possible_votes = max_color_votes + max_motion_votes
            positiveVotes = np.append(motion_votes, color_votes)

            label_M_C = 1
            if len(positiveVotes) > 1:
                label_from_votes, number_of_votes = stats.mode(positiveVotes)
                ratio_agreement_in_color_votes = \
                    float(number_of_votes) / total_possible_votes
                if ratio_agreement_in_color_votes >= self.accept_th:
                    label_M_C = label_from_votes

            y_pred_M_C = np.append(y_pred_M_C, label_M_C)

        return y_pred_M_C
