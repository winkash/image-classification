import numpy as np
import os
import shutil
import pickle
import cv2

from logging import getLogger
from collections import defaultdict
from affine.detection.utils.bounding_box_utils import write_to_dir
from affine.detection.vision.utils.scene_functions import get_config
from affine.video_processing import image_path_to_time, sample_images
from affine.model import Box, session
from affine.model.classifier_models import CaffeModel
from affine.detection.model_worker.tasks.data_processor_task import \
    DataProcessorClient, convert_files_to_bin

logger = getLogger(__name__)

VTEXT_CFG_SPEC = """
[data_params]
    max_imgs = integer(min=1, max=50000, default=50)
    model_id = integer(min=1, default=4)
    pred_th = float(min=0, max=1, default=0.2)
    score_th = float(min=0, max=1, default=0.3)
"""


class VisionTextRecognizer(object):

    LEXICON = 'lexicon.npy'
    CONFIG_FILE = 'Configfile.cfg'
    PATCH_W = 100
    PATCH_H = 32
    STD_VAL = 128.0

    def __init__(self, configfile_name):
        """
        Load and initialize all the configuration params

        Args:
            configfile_name: string with full-path name of the config file
        """
        assert os.path.exists(configfile_name), \
            'Config file %s does not exist' % (configfile_name)

        correct_param_load = self.load_config_file(configfile_name)
        assert correct_param_load,\
            'Config params could not be loaded from file'

        self.configfile = configfile_name
        self.word_rec_cnn = None
        self.img_files = None
        self.lexicon = None

    def load_config_file(self, configfile_name):
        params_loaded = False
        cfg_obj = get_config(configfile_name, spec=VTEXT_CFG_SPEC.split('\n'))
        if cfg_obj:
            self.max_imgs = cfg_obj['data_params']['max_imgs']
            self.model_name = cfg_obj['data_params']['model_name']
            cm = CaffeModel.by_name(self.model_name)
            assert cm,\
                "%s model, from Config file, does not exist in the DB" % \
                (self.model_name)
            logger.info('Using %s'
                        % (self.model_name))
            self.pred_th = cfg_obj['data_params']['pred_th']
            self.score_th = cfg_obj['data_params']['score_th']
            params_loaded = True
        return params_loaded

    def load_model(self, model_dir):
        """
        Load previously trained classifier model.  Loads config file with
        general configuration of the classifier and loads the word rec cnn

        Args:
            model_dir: directory which contains:
                config_file:full-path name to configuration filename
                    Default is CFG_FILE = "Configfile.cfg"
                model_file: lexicon.npy

        Raises/Assertions:
            Asserts if coeffs file, config_file, or model file does not exist.
            Asserts if the config_file fails the formatting check.
        """
        assert os.path.exists(model_dir), \
            "Folder %s with model files does not exist" % (model_dir)

        config_file = os.path.join(model_dir, self.CONFIG_FILE)
        assert os.path.exists(config_file), \
            "Config file not found in model folder %s" % (model_dir)

        cfg_loaded_ok = self.load_config_file(config_file)
        assert cfg_loaded_ok, 'Config file params could not be loaded'

        self.word_rec_cnn = DataProcessorClient(self.model_name)
        self.configfile = config_file
        lexicon_file = os.path.join(model_dir, self.LEXICON)
        assert os.path.exists(lexicon_file), \
            "Lexicon file not found in model folder %s" % (model_dir)

        self.lexicon = np.load(lexicon_file)

    def save_model(self, output_dir):
        """
        Save model when train_model is implemented.

        Args:
            output_dir: folder where we want to save the model files.
                It is expected to be an EMPTY FOLDER
                or to not exist (in this case it will be created)
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info('Saving model')
        dst_config_file = os.path.join(output_dir, self.CONFIG_FILE)
        if self.configfile != dst_config_file:
            shutil.copy(self.configfile, dst_config_file)

        pickle.dump(self.lexicon,
                    open(os.path.join(output_dir, self.LEXICON), 'wb'))

    def train(self, data_dir):
        """
        Call train model on CaffeModel when it is implemented.
        """
        return 0

    def test(self, images, boxes):
        """
        Grab patches from images using boxes, normalize, and predict.
            Args:
                images: dict of timestamp to image file
                boxes: list of tuples (idx, h, w, x, y)
            Returns:
                words: list of words for each patch
        """
        patch_files = []
        words = []
        for timestamp, im_file in images.iteritems():
            image = cv2.imread(im_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            patches = []
            for b in boxes[timestamp]:
                h, w, x, y = b
                img = image[x:x+h, y:y+w]
                m, n = img.shape
                mean = np.mean(img)
                std = np.std(img)
                img = (img - mean) / (std / self.STD_VAL)
                img = cv2.resize(img, (self.PATCH_W, self.PATCH_H))
                img = img.reshape((self.PATCH_H, self.PATCH_W, 1))
                patches.append(img)
            patch_files += write_to_dir(patches)
        prediction = self.word_rec_cnn.predict(convert_files_to_bin(patch_files))
        inds = np.argmax(prediction, axis=1)
        for num, ind in enumerate(inds):
            prob = prediction[num][ind]
            if prob > self.pred_th:
                words.append(self.lexicon[ind])
            else:
                words.append('')
        return words

    def recognize_text_in_video(self, image_dir, box_ids):
        """
        Returns words for each text box in the video.  Only
        accepts a max number of images.  If there are too many images we only
        evaluate up to max_imgs and log that max images was reached.

        Args:
            image_dir: string path to images
            box_ids: list of box ids
        """
        images = sample_images(image_dir, self.max_imgs, 0)
        images = dict((image_path_to_time(im), im) for im in images)
        boxes = session.query(
            Box.timestamp, Box.height, Box.width, Box.y, Box.x, Box.id).filter(
            Box.id.in_(box_ids)).order_by(Box.id.asc()).all()
        all_box_ids = []
        filtered_boxes = defaultdict(list)
        for b in boxes:
            timestamp, h, w, x, y, b_id = b
            if timestamp in images:
                filtered_boxes[timestamp].append([h, w, x, y])
                all_box_ids.append(b_id)
            else:
                logger.info('Image %s not in dir %s' % (timestamp, image_dir))
        words = self.test(images, filtered_boxes)
        box_dict = {}
        for idx, b_id in enumerate(sorted(all_box_ids)):
            box_dict[b_id] = words[idx]
        return box_dict
