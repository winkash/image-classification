import os
import shutil
import tempfile

from affine.model import *
from affine.video_processing import image_path_to_time, sample_images
from .spatial_scene_classifier import SpatialSceneClassifier
from ..utils.scene_functions import POS_LABEL

DEFAULT_IMAGES_PER_VIDEO = 50 # we should have this one in the config
MIN_IMAGES_PER_VIDEO = 1

class NotEnoughImages(Exception):
    """Too few images to make a judgment about the video"""

def judge_images(image_dir, model_dir, image_threshold):
    image_paths = sample_images(image_dir, DEFAULT_IMAGES_PER_VIDEO, MIN_IMAGES_PER_VIDEO)
    if image_paths is None:
        raise NotEnoughImages
    image_paths = [os.path.abspath(x) for x in image_paths]
    timestamps = [image_path_to_time(x) for x in image_paths]
    clf = SpatialSceneClassifier.load_from_dir(model_dir)
    labels = clf.test(image_paths, svm_threshold=image_threshold)
    results = [True if x == POS_LABEL else False for x in labels]
    return dict(zip(timestamps, results))

def judge_video(image_dir, model_dir, video_threshold, image_threshold):
    try:
        image_results = judge_images(image_dir, model_dir, image_threshold)
    except NotEnoughImages:
        return {}, False

    video_result = (image_results.values().count(True) >= video_threshold)
    return image_results, video_result
