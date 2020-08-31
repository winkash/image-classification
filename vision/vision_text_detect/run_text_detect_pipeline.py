import os
import tempfile

from logging import getLogger

from affine.detection.vision.vision_text_detect.vision_text_detector import \
    VisionTextDetector
from affine.detection.vision.vision_text_detect.inject_detector_to_db import \
    TextDetectClassifierInjector


WORD_DET_DATASET = 'word_det_dataset.pkl.gz'
BB_REG_DATASET = 'bb_reg_dataset.pkl.gz'
logger = getLogger(__name__)


def inject_classifier(model_dir, target_label_id, detector_name,
                      pred_thresh=None):
    """Creates VisionText injector object and injects the detector results

    Args:
        model_dir: path to folder with model files
        target_label_id: long
        detector_name: string
        pred_thresh: float

    Returns:
        det, detector object created and injected on S3

    Raises/Assertions:

    """
    di = TextDetectClassifierInjector(model_dir)
    det = di.inject_detector(
        detector_name=detector_name, label_id=target_label_id,
        pred_thresh=pred_thresh)

    return det


def run_pipeline(configfile_name, input_dir, output_dir, detector_name,
                 target_label_id, num_estimators=10):
    """Train and validate a vision text classifier using the given training
    data, and create and inject the corresponding detector if the oob score
    is higher than the threshold

    Args:
        input_dir: string, path to the folder where to find the training sets
            (or to download them if they are not there)
            - RFC training set [REQUIRED] is formatted as:
                'word_det_dataset.pkl.gz'
                Dataset is formatted as tuple of X, y
                where X is a list of features
            - bb_reg training set [OPTIONAL] is formatted as
            'bb_reg_dataset.pkl.gz', a tuple of input_images, p_boxes, g_boxes
            NOTE: If there is no bb_regresion,
            bb_regress_params.model_name should be None

        configfile_name: string, path to find the config file (Configfile.cfg)
        detector_name: string. Note that the name of the detectors is unique,
            they won't be over-written in the DB if they already exist
        target_label_id: int, label_id that we want to target with the detector
        num_estimators: number of trees to use in random forest classifier

    Returns:
        detector object

    Raises/Assertions:
        AssertionError: get_config raises AssertionException if configfile
            has bad formatting: 'Config file validation failed'
    """
    logger.info("Initialize classifier and load config")
    model_dir = tempfile.mkdtemp()
    vt_detector = VisionTextDetector(configfile_name=configfile_name)

    logger.info("starting training pipeline")
    rfc_score = 1
    wd_data_dir = os.path.join(input_dir, WORD_DET_DATASET)
    bb_reg_scores = [1]
    bb_data_dir = os.path.join(input_dir, BB_REG_DATASET)
    bb_regresion_enabled = False
    if vt_detector.bb_reg_model_name:
        bb_regresion_enabled = True
    assert os.path.exists(bb_data_dir) == bb_regresion_enabled, \
        "If bb regresion is disabled in configfile, we should not have training input for it (and viceversa)"

    logger.info("Training RF clf")
    if os.path.exists(wd_data_dir):
        rfc_args = {'data': wd_data_dir, 'num_estimators': num_estimators}
        rfc_score = vt_detector.train(vt_detector.WORD_DET_RFC, rfc_args)

    if bb_regresion_enabled:
        logger.info("Training BB regresion")
        bb_args = {'data': bb_data_dir}
        bb_reg_scores = vt_detector.train(vt_detector.REGRESSION_PARAMS,
                                          bb_args)

    det = None
    if rfc_score > vt_detector.rfc_score_th and \
            min(bb_reg_scores) > vt_detector.bb_score_th:
        logger.info("Saving model")
        vt_detector.save_model(output_dir)
        logger.info("Inject detector")
        det = inject_classifier(model_dir, target_label_id, detector_name)
        logger.info('\t Detector injected %s :' % (det))
    else:
        logger.error(
            'Training results are not good enough to store this classifier')
    return det
