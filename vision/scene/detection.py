import os
import shutil
import tempfile

from affine.model import *
from affine.video_processing import image_path_to_time, sample_images
from .recognition.scene_classifier import SceneClassifier

DEFAULT_IMAGES_PER_VIDEO = 50 # we should have this one in the config
MIN_IMAGES_PER_VIDEO = 1

class NotEnoughImages(Exception):
    """Too few images to make a judgment about the video"""


def make_infofile(outfile, imagedir, default_images, min_images):
    """Write infofile for bovw that contains sampling of images from a directory"""
    images = sample_images(imagedir, default_images, min_images)
    if not images:
        raise NotEnoughImages
    for path in images:
        path = os.path.abspath(path)
        timestamp = image_path_to_time(path)
        line = '%s 1 1 %s\n' % (path, timestamp)
        outfile.write(line)


def judge_images(imagedir, model_dir):
    """Inner function to calculate scene detection confidence of an image
        params is a dictionary with info about the paths of files """
    intermediate = tempfile.mkdtemp()
    try:
        info_file_path = os.path.join(intermediate, 'info.txt')
        with file(info_file_path, 'w') as infofile:
            make_infofile(infofile,
                          imagedir,
                          DEFAULT_IMAGES_PER_VIDEO,
                          MIN_IMAGES_PER_VIDEO) #create info file

        config_path = os.path.join(model_dir, 'params.cfg')
        scene = SceneClassifier(config_path, intermediate)
        scene.set_model_files(model_dir,
                              info_file_path,
                              os.path.join(model_dir, 'pca.xml'),
                              os.path.join(model_dir, 'vocab.xml'),
                              os.path.join(model_dir, 'model.svm'))
        # the test_classifier returns results as a dict
        # key : (video_id, timestamp),
        # value : svm result for corresponding tuple
        results = scene.test_classifier()
        image_results = {}
        for (_, timestamp), score in results.items():
            image_results[timestamp] = score

        return image_results

    finally:
        shutil.rmtree(intermediate)


def judge_video(imagedir, model_dir, image_threshold, video_threshold):
    """Compute final veredict for a video"""
    try:
        image_scores = judge_images(imagedir, model_dir)
    except NotEnoughImages:
        return {}, False

    num_hits = 0
    image_results = {}
    for image, score in image_scores.iteritems():
        result = (score >= image_threshold)
        image_results[image] = result
        num_hits += int(result)
    #now we check how many times we saw this image in the video
    #video threshold is the number of scenes required to give a VDR
    video_result = num_hits >= video_threshold
    return image_results, video_result
