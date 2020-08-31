import argparse
import os
from logging import getLogger
from configobj import ConfigObj
from validate import Validator

from affine.detection.model.features import SurfExtractor, BagOfWords
from affine.detection.vision.utils import scene_functions
from affine.detection.vision.logo_recognition.bow_training import train_bag_of_words
from affine.detection.vision.logo_recognition.knn_training import KnnTrainer
from affine.detection.vision.logo_recognition.logo import Logo

logger = getLogger(name=__name__)

CFG_SPEC = """
logo_dir = string
op_model_dir = string
[BOF]
    patch_shapes = string
    scales = string
    step_size = float(default=0.5)
    center_area_offset = integer(default=None)
    corner_area_sz = string(default=None)
    contrast_thresh = integer(default=None)
    variance_thresh = integer(default=None)
    raise_on_size = boolean(default=False)
[size]
    resize = boolean(default=False)
    standard_width = integer(default=100)
    max_logo_size = integer(default=120)
[BoW]
    train_bow = boolean(default=True)
    bow_dir = string(default="")
    vocabsize = integer
    num_train_images = integer
[SURF]
    hessianThreshold = float
    keypoint_limit = integer
[KNN]
    k_neighbors = integer
    metric = string(default="manhattan")
[RbM]
    min_points = integer
    min_matches = integer
    ransac_th = float(default=0.5)
    accept_th = float(default=0.3)
    ransac_algorithm = integer(default=8)
    ransac_max_iter = integer(default=50)
    ransac_prob = float(min=0.0, max=1.0, default=0.95)
    inlier_r = float(min=0.0, max=1.0, default=0.5)
"""


def train(config_file):
    """The method that wraps around all steps required to train
    a new Logo recognition model

    Args:
        config_file : The path to the config file

    It save all the files in the op_model_dir mentioned in the config_file
    """
    assert os.path.exists(config_file), "Invalid Config file"

    logger.info("Validating Config File")
    cfg = scene_functions.get_config(config_file, CFG_SPEC.split('\n'))

    logo_dir = cfg['logo_dir']
    assert os.path.isdir(logo_dir), "Logo dir does not exist !"

    op_model_dir = cfg['op_model_dir']
    if not os.path.isdir(op_model_dir):
        os.makedirs(op_model_dir)

    # train BoW
    train_bow = cfg['BoW']['train_bow']
    if train_bow:
        logger.info("Training BoW")
        hessian_threshold = cfg['SURF']['hessianThreshold']
        keypoint_limit = cfg['SURF']['keypoint_limit']
        feat_ext = SurfExtractor(
            hessian_thresh=hessian_threshold, keypoint_limit=keypoint_limit)
        logo_paths = [os.path.join(logo_dir, img) for img in os.listdir(logo_dir)]
        vocabsize = cfg['BoW']['vocabsize']
        num_images = cfg['BoW']['num_train_images']
        bow, _ = train_bag_of_words(vocabsize,
                                    feat_ext,
                                    logo_paths,
                                    num_images=num_images)
    else:
        logger.info("Using pre-trained bow model")
        bow_dir = cfg['BoW']['bow_dir']
        assert os.path.isdir(bow_dir), "BoW dir does not exist"
        bow = BagOfWords.load_from_dir(bow_dir)

    # train KNN
    logger.info("Training KNN")
    k_neighbors = cfg['KNN']['k_neighbors']
    metric = cfg['KNN']['metric']
    knn = KnnTrainer(bow, neighbors=k_neighbors, metric=metric)
    max_logo_size = cfg['size']['max_logo_size']
    logos = Logo.load_from_dir(logo_dir, max_logo_size=None)
    knn.train(logos)

    # saving all the trained files
    logger.info("Saving all to %s", op_model_dir)
    knn.save_knn(op_model_dir)
    knn.save_logos(op_model_dir)
    bow.save_to_dir(os.path.join(op_model_dir, 'bow'))

    # saving the config file
    cfg.filename = os.path.join(op_model_dir, "model.cfg")
    cfg.write()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file')
    args = parser.parse_args()
    train(args.cfg_file)


if __name__ == '__main__':
    main()
