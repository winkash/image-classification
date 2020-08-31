import subprocess

import os
import socket

import signal

import numpy as np
from logging import getLogger

from affine.detection.model.features import NerFeatureExtractor
from affine.model import NerDetector
from affine.detection.model.classifiers import LibsvmClassifier

logger = getLogger(__name__)


def process_page(page, detectors):
    """ Runs NER classification on a page"""
    logger.info("Running NER detection on page %d"%page.id)
    nfe = NerFeatureExtractor()
    try:
        ftr_dict = nfe.extract(page.id)
    except socket.timeout:
        logger.exception("Skipping NER due to timeout")
        # Kill server to recover from bad state.
        # The server should be started automatically for next detection
        _kill_ner_server()
        return

    matching_detectors = set()
    for det in detectors:
        ftrs = ftr_dict.get(det.clf_target.target_label_id)
        if ftrs is not None:
            if classify_ftrs(ftrs, det):
                logger.info("NER true detection (page_id:%d, detector:%s)"%(page.id, det.name))
                det.save_result(page.id)
                matching_detectors.add(det)
    detectors_to_delete = set(detectors) - matching_detectors
    detector_ids_to_delete = [detector.id for detector in detectors_to_delete]
    NerDetector.delete_detector_results(page, detector_ids_to_delete)


def classify_ftrs(ftrs, det):
    det.grab_files()
    model_file = det.local_path(NerDetector.SVM_MODEL)
    clf = LibsvmClassifier.load_from_file(model_file)
    return clf.predict(np.asarray([ftrs]))[0]


def _kill_ner_server():
    logger.info('Killing NER server because it is in a bad state')
    proc = subprocess.Popen(["pgrep", '-f', 'NERServer'], stdout=subprocess.PIPE)
    for pid in proc.stdout:
        os.kill(int(pid), signal.SIGKILL)
