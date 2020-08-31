import os
import random

from affine.model.detection import ResultAggregator
from affine.video_processing import image_path_to_time, sample_images
from affine.detection.vision.utils.scene_functions import get_config
from .logo_client import LogoClient
from .injection import CFG_FILE_NAME, CFG_SPEC

__all__ = ['judge_video']

SAMPLE_IMAGE_COUNT = 50
IMAGES_PER_VIDEO = 10


def judge_video(clf_model_dir, imagedir):
    image_paths = sample_images(imagedir, SAMPLE_IMAGE_COUNT)
    random.shuffle(image_paths)
    image_paths = image_paths[:IMAGES_PER_VIDEO]
    model_name = get_model_name(clf_model_dir)
    logo_client = LogoClient(model_name)
    image_results = logo_client.predict(image_paths)
    timestamps = map(image_path_to_time, image_paths)
    ra = ResultAggregator()
    for ts, labeled_boxes in zip(timestamps, image_results):
        for h, w, y, x, target_label_id in labeled_boxes:
            ra.add_new_box(x, y, w, h, ts, 'Logo', label_id=target_label_id)
    return ra.result_dict


def get_model_name(clf_model_dir):
    cfg_path = os.path.join(clf_model_dir, CFG_FILE_NAME)
    cfg = get_config(cfg_path, CFG_SPEC.split('\n'))
    return cfg['model_name']
