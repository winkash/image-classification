import os

from logging import getLogger
from shutil import rmtree
from tempfile import mkdtemp

from affine.detection.vision.utils.scene_functions import download_images_to_dir
from affine.detection.vision.spatial_scene.inject import SpatialSceneDetectorInjector
from affine.detection.vision.spatial_scene.spatial_scene_classifier import \
        SpatialSceneClassifier

__all__ = ['evaluate_and_inject', 'PRECISION_THRESHOLD',
           'RECALL_THRESHOLD']

logger = getLogger(__name__)
 
PRECISION_THRESHOLD = 0.8
RECALL_THRESHOLD = 0.8


def evaluate_and_inject(clf, images, labels, precision_thrsh, recall_thrsh,
                        det_name, target_label, video_threshold):
    """Evaluate classifier using cross-validation and inject to DB if good.
    
    Args:
        clf: Spatial scene classifier.
        images: List of (video id, timestamp)s.
        labels: List of labels.
        precision_thrsh: Precision threshold for cross-validation.
        recall_thrsh: Recall threshold for cross-validation.
        det_name: Detector name.
        target_label: Detector's target label.
        video_threshold: Detector's video threshold (# images the detector must
                         fire on to produce a VDR).

    Raises:
        AssertionError: Bad args.
        Exception: Precision/Recall requirements not met.
    """
    assert isinstance(clf, SpatialSceneClassifier)
    assert 0 <= precision_thrsh <= 1
    assert 0 <= recall_thrsh <= 1
    assert video_threshold > 0
    logger.info("Downloading images...")
    image_dir = mkdtemp()
    try:
        image_paths = download_images_to_dir(images, image_dir)
        logger.info("Evaluating...")
        precision, recall = clf.evaluate(image_paths, labels)
        logger.info("Precision = %s, Recall = %s" % (precision, recall))
        if not (precision >= precision_thrsh and recall >= recall_thrsh):
            raise Exception("Precision/Recall requirements not met!")
        logger.info("Training...")
        clf.train(image_paths, labels)
        model_dir = mkdtemp()
        try:
            clf.save_to_dir(model_dir)
            inj = SpatialSceneDetectorInjector(model_dir)
            logger.info("Injecting...")
            det = inj.inject(det_name, target_label, video_threshold, (images, labels))
            logger.info('Injection successful! Detector id = %s' % det.id)
        finally:
            rmtree(model_dir)
    finally:
        rmtree(image_dir)
