import os
import shutil
import pickle
import gzip
import cv2

from logging import getLogger
from sklearn.ensemble import RandomForestClassifier

from affine.detection.utils.bounding_box_utils import write_to_dir
from affine.detection.vision.utils.scene_functions import get_config
from affine.detection.model import EdgeBoxExtractor, GridBoxExtractor,\
    BoundingBoxRegressor
from affine.model.classifier_models import CaffeModel
from affine.video_processing import sample_images
from affine.detection.model_worker.tasks.data_processor_task import \
    DataProcessorClient, convert_files_to_bin
from affine.detection.model_worker.utils import merge_list, batch_list

logger = getLogger(__name__)

# Setup box extractor in the spec
DEFAULT_BOX_EXT = 'edge_box_params'
BOX_EXT_OPTIONS = ['grid_box_params', DEFAULT_BOX_EXT]
BOX_EXT_OPTIONS.extend([DEFAULT_BOX_EXT])

VTEXT_CFG_SPEC = """
[data_params]
    max_img_size = float(min=10., default=1024.)
    max_imgs = integer(min=1, max=50000, default=50)
    rfc_score_th = float(min=0, max=1, default=.9)
    bb_score_th = float(min=0, max=1, default=.5)
    box_extractor = """ \
    """option('{}', '{}', default='{}')
    """.format(*BOX_EXT_OPTIONS) +\
"""[grid_box_params]
    grid_shape = int_list(default=list(2, 2))
[edge_box_params]
    patch_widths = int_list(default=list(32, 64, 100, 150))
    scales = float_list(default=list(1, .5))
    alpha = float(min=0, max=1, default=.65)
    min_group_size = integer(min=0, default=8)
    score_thresh = float(min=0, default=5)
    mag_th = float(min=0, max=1, default=0.1)
[word_det_params]
    img_size = int_list(default=list(32, 100))
    pred_thresh = float(min=0, max=1, default=0.7)
    model_name = string(min=0, max=256, default='rfc')
[bb_regress_params]
    model_name = string(min=0, max=256, default=None)
"""


class VisionTextDetector(object):
    """ Text detector pipeline that takes in the path to an image, finds
    bounding boxes of text, and returns the characters in those boxes.
    Has functionality for finding text in multiple neighboring frames
    """

    CONFIG_FILE = 'Configfile.cfg'
    WORD_DET_RFC = 'word_det_rfc.pickle'
    REGRESSION_PARAMS = 'reg_coeffs.pickle'

    def __init__(self, configfile_name):
        """
        Load and initialize all the configuration params

        Args:
            configfile_name: string with full-path name of the config file

        Returns:

        Raises/Assertions:
        """
        assert os.path.exists(configfile_name), \
            'Config file %s does not exist' % (configfile_name)

        correct_param_load = self.load_config_file(configfile_name)
        assert correct_param_load,\
            'Config params could not be loaded from file'

        self.fullpath_input_configfile = configfile_name
        self.word_det_rfc = None
        self.reg_coeffs = None
        self.bb_reg = None
        self.img_files = None

    def _get_model_id(self, model_name):
        cm = CaffeModel.by_name(model_name)
        assert cm,\
            "%s model, from Config file, does not exist in the DB" % \
            (model_name)
        model_id = cm.id
        logger.info('Using %s, id %d'
                    % (model_name, model_id))
        return model_id

    def load_config_file(self, configfile_name):
        params_loaded = False
        cfg_obj = get_config(configfile_name, spec=VTEXT_CFG_SPEC.split('\n'))
        if cfg_obj:
            box_extractor = cfg_obj['data_params']['box_extractor']
            self.max_img_size = cfg_obj['data_params']['max_img_size']
            self.max_imgs = cfg_obj['data_params']['max_imgs']
            self.rfc_score_th = cfg_obj['data_params']['rfc_score_th']
            self.bb_score_th = cfg_obj['data_params']['bb_score_th']

            wd_dict = cfg_obj['word_det_params']
            self.word_img_size = tuple(wd_dict['img_size'])
            self.pred_thresh = wd_dict['pred_thresh']
            self.word_det_model_name = wd_dict['model_name']
            cm = CaffeModel.by_name(self.word_det_model_name)
            assert cm,\
                "%s model, from Config file, does not exist in the DB" % \
                (self.word_det_model_name)
            self.box_ext_kwargs = cfg_obj[box_extractor]
            if box_extractor == 'edge_box_params':
                self.box_ext_cls = EdgeBoxExtractor
                patch_widths = self.box_ext_kwargs.pop('patch_widths')
                patch_shapes = [(self.word_img_size[0], w) for w in patch_widths]
                self.box_ext_kwargs['patch_shapes'] = patch_shapes
            elif box_extractor == 'grid_box_params':
                self.box_ext_cls = GridBoxExtractor
            logger.info('Using %s' % (self.word_det_model_name))

            bb_dict = cfg_obj['bb_regress_params']
            self.bb_reg_model_name = bb_dict['model_name']
            if self.bb_reg_model_name:
                cm = CaffeModel.by_name(self.bb_reg_model_name)
                assert cm,\
                    "%s model, from Config file, does not exist in the DB" % \
                    (self.bb_reg_model_name)

            params_loaded = True
        return params_loaded

    def load_model(self, model_dir):
        """
        Load a previously trained classifier model.

        This function loads the config file with general configuration of the
        classifier and loads the random forest classifier, regression model,
        and r-cnn.

        Args:
            model_dir: directory which contains:
                config_file: string, full-path name to configuration filename.
                    Default is CFG_FILE = "Configfile.cfg"
                model_files:
                    "word_detection_rfc.pickle"
                    "reg_coeffs.pickle"

        Returns:

        Raises/Assertions:
            Asserts if coeffs file, config_file, or model file does not exist.
            Asserts if the config_file fails the formatting check.
        """
        assert os.path.exists(model_dir), \
            "Folder %s with model files does not exist" % (model_dir)

        config_file = os.path.join(model_dir, self.CONFIG_FILE)
        assert os.path.exists(config_file), \
            "Config file not found in model folder %s" % (model_dir)

        rfc_file = os.path.join(model_dir, self.WORD_DET_RFC)
        assert os.path.exists(rfc_file), \
            "RFC pickle file not found in model folder %s" % (model_dir)

        coeffs_file = os.path.join(model_dir, self.REGRESSION_PARAMS)
        assert os.path.exists(coeffs_file), \
            "Coefficients file not found in model folder %s" % (model_dir)

        cfg_loaded_ok = self.load_config_file(config_file)
        assert cfg_loaded_ok, 'Config file params could not be loaded'

        self.word_det_rfc = pickle.load(open(rfc_file))
        self.reg_coeffs = pickle.load(open(coeffs_file))
        self.word_det_cnn = DataProcessorClient(self.word_det_model_name)
        self.fullpath_input_configfile = config_file
        if self.bb_reg_model_name:
            self.bb_reg = BoundingBoxRegressor(self.bb_reg_model_name)

    def save_model(self, output_dir):
        """
        Save current trained classifier model.

        This function stores two pickle files, and a copy of the config file
            used for the current classifier as:
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

        logger.info('Saving model')
        dst_config_file = os.path.join(output_dir, self.CONFIG_FILE)
        if self.fullpath_input_configfile != dst_config_file:
            shutil.copy(self.fullpath_input_configfile, dst_config_file)

        pickle.dump(self.word_det_rfc,
                    open(os.path.join(output_dir, self.WORD_DET_RFC), 'wb'))
        pickle.dump(self.reg_coeffs, open(
            os.path.join(output_dir, self.REGRESSION_PARAMS), 'wb'))

    def train(self, model, args):
        """
        Calls specific training functions based on input
        Args:
            model: self.WORD_DET_RFC or self.REGRESSION_PARAMS
            args: dict of parameters for model training
        Returns: score (float)
        """
        if model == self.WORD_DET_RFC:
            return self.train_rfc(args)
        elif model == self.REGRESSION_PARAMS:
            return self.train_bb_reg(args)
        else:
            raise Exception('No model %s exists to train' % model)

    def train_rfc(self, data, num_estimators=10, max_depth=64, oob_score=True):
        """
        Trains Random Forest Classifier with parameters and training data
        Args:
            data: string path to pickled and tarred data file with training
            data for the random forest classifier formatted as a tuple: X, y.
            num_estimators: number of trees in the random forest classifier

        Returns: Out of Bag Score

        Raises/Assertions:
            Assert that data file exists.
        """
        logger.info('Training RFC model')
        assert os.path.exists(data), "File %s with data does not "\
            "exist" % (data)
        X, y = pickle.load(gzip.open(data))
        self.word_det_rfc = RandomForestClassifier(
            n_estimators=num_estimators, max_depth=max_depth,
            oob_score=oob_score)
        self.word_det_rfc.fit(X, y)
        return self.word_det_rfc.oob_score_

    def train_bb_reg(self, data):
        """
        Trains bounding box regression model using features extracted from
        R-CNN.
        Args: string path to pickled and tarred data file for input_images,
            p_boxes, and g_boxes for BoundingBoxRegressor.train_model

        Returns: Averaged cross validation scores for h, w, x, y models
        """
        logger.info('Training Bounding Box Regression model')
        assert os.path.exists(data), "File %s with data does not "\
            "exist" % (data)
        input_images, p_boxes, g_boxes = pickle.load(gzip.open(data))
        self.bb_reg = BoundingBoxRegressor(self.bb_reg_model_name)
        coefficients, scores = self.bb_reg.train_model(
            input_images, p_boxes, g_boxes)
        self.reg_coeffs = coefficients
        return scores

    def test(self):
        """
        Runs edge boxes to extract candidate bounding boxes, then runs them
        through a random forest classifier, then a regression model to refit
        bounding boxes.

        Returns:
            patch_map: list of bounding boxes and corresponding words.
                Each bb is idx, h, w, x, y.
                If no patches are returned, return empty list.
        """
        ebe = self.box_ext_cls(**self.box_ext_kwargs)
        edge_boxes = ebe.extract(self.img_files)
        patches, patch_map = self.convert_edge_boxes(edge_boxes)
        if patches:
            patch_files = write_to_dir(patches)
            batches = batch_list(patch_files, batch_size=1)
            features = []
            for batch in batches:
                features.append(
                    self.word_det_cnn.predict(convert_files_to_bin(batch)))
            features = merge_list(features)
            pred = self.word_det_rfc.predict_proba(features)
            patch_map = [pm for idx, pm in enumerate(patch_map)
                         if pred[idx][1] > self.pred_thresh]
            if patch_map and self.bb_reg:
                patch_map = self.bb_reg.extract(
                    self.reg_coeffs, patch_map, self.img_files)
        return patch_map

    def detect_text_in_video(self, image_dir):
        """Returns bounding rectangles for each frame in the video.  Only
        accepts a max number of images.  If there are too many images we only
        evaluate up to max_imgs and log that max images was reached.

        Args:
            frames: list of imgs from video, assuming they end in '.jpg'

        Returns:
            img_files: list of img files
            bounding_rectslist of bounding rectangles
        """
        self.img_files = sample_images(image_dir, self.max_imgs, 0)

        bounding_rects = self.test()
        return self.img_files, bounding_rects

    def convert_edge_boxes(self, edge_boxes):
        """Convert edge_boxes format to patches and patch_map, and resize
        to word_img_size.

        Params: edge_boxes: dict of img_size and list of tuples (h, w, x, y)
        Returns: patches: list of flattened patches
            patch_map: list of img index, patch size, and patch upper left
                corner
        """
        patches = []
        patch_map = []
        for i, image in enumerate(self.img_files):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img is not None and i in edge_boxes:
                boxes = edge_boxes[i]
                for b in boxes:
                    h, w, x, y = b
                    patch = img[x:x + h, y:y + w]
                    patch = cv2.resize(patch, (self.word_img_size[1],
                                               self.word_img_size[0]))
                    patches.append(patch)
                    patch_map.append((i, h, w, x, y))
        return patches, patch_map
