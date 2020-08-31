from logging import getLogger
from scipy.sparse import csr_matrix
import numpy as np
import os
import tempfile

from affine import config
from affine.model import LdaDetector
from ..topic_model import *
from .lda_client import LdaClient

logger = getLogger(__name__)

INFER_LDA_JAR = os.path.join(config.bin_dir(), 'topic_model', 'InferLDA.jar')
lda_model_lookup = {}

def temp_path():
    fd, path = tempfile.mkstemp()
    os.close(fd)
    return path


def process_page(page, detectors):
    """Run lda detectors on webpage text"""
    logger.info("Running LDA detection on page %d", page.id)
    global lda_model_lookup
    lda_model_lookup = {}

    detectors = set(detectors)
    detector_ids_to_delete = set()

    for detector in detectors:
        logger.info('Running LDA detection (page_id:%d detector:%s)', page.id, detector.name)
        if classify_text(page.title_and_text, detector):
            logger.info("LDA true detection (page_id:%d, detector:%s)", page.id, detector.name)
            detector.save_result(page.id)
        else:
            detector_ids_to_delete.add(detector.id)

    LdaDetector.delete_detector_results(page, detector_ids_to_delete)
    logger.info("Finished LDA detection on page %d", page.id)


def classify_text(text, det):
    det.grab_files()
    cfg_file = det.local_path(PipelineRunner.CFG_NAME)
    config_obj = PipelineRunner.validate_config_file(cfg_file)
    vocab_file = det.local_path(config_obj['vocab_file'])
    with open(vocab_file) as fi:
        vocab_set = set(fi.read().decode('utf-8').splitlines())
    clean_text = TopicTrainer.preprocess_text(text, vocab_set)
    if not clean_text:
        return 0
    topic_dist_sparse = mallet_infer_topics(clean_text, det, config_obj)
    pred = classify_topic_distribution(topic_dist_sparse, det, config_obj)
    return pred


def classify_topic_distribution(topic_dist_mat, det, config_obj):
    prediction_file = temp_path()
    topic_thresholds = config_obj['topic_thresholds']
    classifier_model = det.local_path(config_obj['classifier_params']['model_file'])
    if len(topic_thresholds):
        TopicTrainer.manual_prediction(topic_dist_mat, topic_thresholds, prediction_file)
    else:
        TopicTrainer.model_prediction(topic_dist_mat, classifier_model, prediction_file)

    with open(prediction_file) as fi:
        pred = int(fi.read().strip())
    assert pred in (0,1)
    return pred


def mallet_infer_topics(clean_text, det, config_obj):
    topic_dist = memoized_infer_topics(clean_text, det.lda_model_id)
    n_ftrs = config_obj['mallet_train']['num-topics']
    sparse_mat = topic_dist_to_sparse(topic_dist, n_ftrs)
    return sparse_mat


def memoized_infer_topics(clean_text, lda_model_id):
    global lda_model_lookup
    try:
        topic_dist = lda_model_lookup[lda_model_id]
    except KeyError:
        topic_dist = LdaClient.infer_topics(clean_text, lda_model_id)
        lda_model_lookup[lda_model_id] = topic_dist
    return topic_dist


def topic_dist_to_sparse(topic_dist, n_ftrs):
    topics, vals = zip(*topic_dist.items())
    return csr_matrix((vals, (np.zeros(len(topics), dtype='int8'), topics)),
            shape=(1, n_ftrs), dtype='float64')
