from logging import getLogger
from argparse import ArgumentParser

from affine.model import SpatialSceneDetector
from affine.detection.vision.spatial_scene.retrain import SpatialSceneRetrainer
from affine.detection.vision.spatial_scene.utils import (PRECISION_THRESHOLD,
                                                         RECALL_THRESHOLD,
                                                         evaluate_and_inject)

logger = getLogger(__name__)
 

def run_retraining_pipeline(old_det_name, new_det_name, precision_thrsh,
                            recall_thrsh):
    """Pipeline for RE-training a spatial scene detector.
    
    Args:
        old_det_name: Name of detector to retrain.
        new_det_name: Name of new detector to be created.
        precision_thrsh: Cross-validation precision that must be achieved.
        recall_thrsh: Cross-validation recall that must be achieved.
    """
    logger.info("Running retraining pipeline...")
    old_det = SpatialSceneDetector.by_name(old_det_name)
    ret = SpatialSceneRetrainer(old_det)
    images, labels = [x + y for (x, y) in zip(ret.get_old_data(),
                                              ret.get_new_data())]
    clf = ret.get_old_classifier()
    evaluate_and_inject(clf, images, labels, precision_thrsh, recall_thrsh,
                        new_det_name, old_det.target_label,
                        old_det.video_threshold)


def parse_cmdline_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('old_det_name', help='name of detector to retrain')
    parser.add_argument('new_det_name', help='name of detector to be created')
    parser.add_argument('--precision_thrsh', type=float, default=PRECISION_THRESHOLD,
                        help='cross-validation precision threshold')
    parser.add_argument('--recall_thrsh', type=float, default=RECALL_THRESHOLD,
                        help='cross-validation recall threshold')
    return parser.parse_args(args=args)
    

def check_args(args):
    assert SpatialSceneDetector.by_name(args.old_det_name)
    assert not SpatialSceneDetector.by_name(args.new_det_name)
    assert 0 <= args.precision_thrsh <= 1
    assert 0 <= args.recall_thrsh <= 1


def main(args=None):
    args = parse_cmdline_args(args=args)
    check_args(args)
    run_retraining_pipeline(args.old_det_name, args.new_det_name,
                            args.precision_thrsh, args.recall_thrsh)


if __name__ == '__main__':
    main()
