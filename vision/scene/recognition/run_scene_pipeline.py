import os
import sys
import argparse

from datetime import datetime
from logging import getLogger

from affine.model import *
from affine.detection.vision.utils.scene_functions \
    import get_config, SCENE_CFG_SPEC
from affine.detection.vision.scene.recognition.scene_classifier \
    import SceneClassifier
from affine.detection.utils import run_mturk_submission, run_url_injection

TPR = 0.8
FPR = 0.1

logger = getLogger(__name__)


def scene_detector(config_filename, inter_dir):
    logger.info("checking config file")
    config_scene = get_config(config_filename, SCENE_CFG_SPEC.split('\n'))
    logger.info("starting training pipeline")
    logger.info("url injection")
    myurls = run_url_injection.query_video_urls(config_scene, inter_dir)
    logger.info("finished url injection")

    logger.info("injected %d urls in the database" % myurls)
    urls = os.path.join(inter_dir, 'all_urls.txt')
    d_start = datetime.now()
    logger.info("checking if videos have been donwloaded")
    n_videos = run_url_injection.check_videos_downloaded(
        urls, config_scene, d_start)
    logger.info("Downloaded %d videos out of %s initial urls" %
                (n_videos, config_scene["url_injection"]["max_vids"]))

    logger.info("Submiting hits to mechanical turk")
    run_mturk_submission.mturk_submission_ingestion(config_scene, inter_dir)
    logger.info("Mturk process finished. At least 95% of answers ingested")
    scene = SceneClassifier(config_filename, inter_dir)
    scene.set_params()

    logger.info("Obtaining positive examples from Mturk")
    posdata = scene.get_positive_examples_mturk()
    logger.info("Training Scene classifier")
    negdata = scene.write_negative_examples()
    logger.info("creating datasets")
    scene.create_dataset_partitions(posdata, negdata)
    logger.info("training")
    scene.train_classifier()
    logger.info("testing")
    results = scene.test_classifier()
    logger.info("Evaluating")
    scene.evaluate_classifier(results)
    if scene.tpr >= TPR and scene.fpr <= FPR:
        scene.inject_classifier()
    else:
        logger.info('Retrain classifier. TPR = %d , FPR = %d' %
                    (scene.tpr, scene.fpr))

if __name__ == '__main__':
    # parsing the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-inter_dir', action='store', dest='inter_dir',
                        required=True, help='set the inter_dir trained detector')

    parser.add_argument('-config', action='store', dest='config_filename',
                        required=True, help='set the config filename')
    results = parser.parse_args()

    if not os.path.exists(results.inter_dir):
        print ' -inter_dir is not found!'
        sys.exit(1)

    if not os.path.exists(results.config_filename):
        print '-c config filename is not found!'
        sys.exit(1)

    scene_detector(results.config_filename, results.inter_dir)
