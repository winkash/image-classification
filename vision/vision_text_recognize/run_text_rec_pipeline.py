import os
import tempfile

from logging import getLogger

from affine.detection.vision.vision_text_recognize.vision_text_recognizer import\
    VisionTextRecognizer
from affine.detection.vision.vision_text_recognize.inject_detector_to_db import \
    TextRecognizeClassifierInjector


logger = getLogger(__name__)

WORD_REC_DATASET = 'word_rec_dataset.pkl.gz'


def inject_classifier(model_dir, target_label_id, detector_name,
                      pred_thresh=None):
    """
    Creates VisionText injector object and injects the detector results

    Args:
        model_dir: path to folder with model files
        target_label_id: long
        detector_name: string
        pred_thresh: float

    Returns:
        det, detector object created and injected on S3
    """
    di = TextRecognizeClassifierInjector(model_dir)
    det = di.inject_detector(
        detector_name=detector_name, label_id=target_label_id,
        pred_thresh=pred_thresh)
    return det


def run_pipeline(configfile_name, input_dir, output_dir, detector_name,
                 target_label_id):
    """
    Train and validate a vision text classifier using the given training
    data, and create and inject the corresponding detector if the oob score
    is higher than the threshold

    Args:
        input_dir:
        configfile_name: string, path to find the config file (Configfile.cfg)
        detector_name: string. Note that the name of the detectors is unique,
            they won't be over-written in the DB if they already exist
        target_label_id: int, label_id that we want to target with the detector

    Returns:
        detector object

    Raises/Assertions:
        AssertionError: get_config raises AssertionException if configfile
            has bad formatting: 'Config file validation failed'
    """
    logger.info("Initialize classifier and load config")
    model_dir = tempfile.mkdtemp()
    vt_recognizer = VisionTextRecognizer(configfile_name=configfile_name)

    logger.info("starting training pipeline")
    data_dir = os.path.join(input_dir, WORD_REC_DATASET)
    score = 0
    if os.path.exists(data_dir):
        score = vt_recognizer.train(data_dir)

    det = None
    if score > vt_recognizer.score_th:
        logger.info("Saving model")
        vt_recognizer.save_model(output_dir)
        logger.info("Inject detector")
        det = inject_classifier(model_dir, target_label_id, detector_name)
        logger.info('\t Detector injected %s :' % (det))
    else:
        logger.error(
            'Training results are not good enough to store this classifier')
    return det
