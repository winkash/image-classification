import os

from logging import getLogger
from argparse import ArgumentParser

from affine.model import Label, SpatialSceneDetector
from affine.detection.vision.spatial_scene.spatial_scene_classifier import SpatialSceneClassifier
from affine.detection.vision.spatial_scene.inject import SpatialSceneDetectorInjector
from affine.detection.vision.spatial_scene.utils import (PRECISION_THRESHOLD,
                                                         RECALL_THRESHOLD,
                                                         evaluate_and_inject)
from affine.detection.vision.utils.scene_functions import (POS_LABEL, NEG_LABEL,
                                                           read_image_file)

logger = getLogger(__name__)
 

def run_training_pipeline(config, pos_file, neg_file, precision_thrsh,
                          recall_thrsh, det_name, target_label_id,
                          video_threshold):
    """Pipeline for training a new spatial scene detector.
    
    Args:
        config: Path to config file.
        pos_file: Path to image file specifying positive images.
        neg_file: Path to image file specifying negative images.
        precision_thrsh: Cross-validation precision that must be achieved.
        recall_thrsh: Cross-validation recall that must be achieved.
        det_name: Name of detector to be created.
        target_label_id: Detector's target label id.
        video_threshold: Detector's video_threshold (the # images the detector
                         must fire on to create a VDR)

    Raises:
        AssertionError: Target label doesn't exist.
    """
    logger.info("Running training pipeline...")
    pos_images = read_image_file(pos_file)
    neg_images = read_image_file(neg_file)
    images = pos_images + neg_images
    labels = [POS_LABEL]*len(pos_images) + [NEG_LABEL]*len(neg_images)
    clf = SpatialSceneClassifier(config)
    target_label = Label.get(target_label_id)
    assert target_label
    evaluate_and_inject(clf, images, labels, precision_thrsh, recall_thrsh,
                        det_name, target_label, video_threshold)


def parse_cmdline_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('pos_file', help='positive image file')
    parser.add_argument('neg_file', help='negative image file')
    parser.add_argument('config', help='config file')
    parser.add_argument('det_name', help='name of detector to be created')
    parser.add_argument('target_label_id', type=int,
                        help="detector's target label id")
    parser.add_argument('video_threshold', type=int,
                        help="detector's video threshold")
    parser.add_argument('--precision_thrsh', type=float,
                        default=PRECISION_THRESHOLD,
                        help='cross-validation precision threshold')
    parser.add_argument('--recall_thrsh', type=float, default=RECALL_THRESHOLD,
                        help='cross-validation recall threshold')
    return parser.parse_args(args=args)
    

def check_args(args):
    assert os.path.exists(args.pos_file)
    assert os.path.exists(args.neg_file)
    assert os.path.exists(args.config)
    assert not SpatialSceneDetector.by_name(args.det_name)
    assert Label.get(args.target_label_id)
    assert args.video_threshold > 0
    assert 0 <= args.precision_thrsh <= 1
    assert 0 <= args.recall_thrsh <= 1
    

def main(args=None):
    args = parse_cmdline_args(args=args)
    check_args(args)
    run_training_pipeline(args.config, args.pos_file, args.neg_file,
                          args.precision_thrsh, args.recall_thrsh,
                          args.det_name, args.target_label_id,
                          args.video_threshold)


if __name__ == '__main__':
    main()
