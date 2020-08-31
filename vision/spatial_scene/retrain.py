from affine.model import TrainingImage
from ...utils.get_training_data import get_image_detector_mturk_images
from .inject import SpatialSceneDetectorInjector
from ..utils.scene_functions import POS_LABEL, NEG_LABEL
from .spatial_scene_classifier import SpatialSceneClassifier

__all__ = ['SpatialSceneRetrainer']

class SpatialSceneRetrainer(object):

    def __init__(self, det):
        self._det = det

    def get_old_data(self):
        ''' Get old detector's training images.

        Returns:
            [(video_id, timestamp), ...], [label, ...]
        '''
        query = TrainingImage.query.filter_by(detector_id=self._det.id)
        images = []
        labels = []
        for ti in query:
            images.append((ti.video_id, ti.timestamp))
            labels.append(ti.label)
        return images, labels

    def get_new_data(self):
        ''' Get MTurk data for old detector.

        Returns:
            [(video_id, timestamp), ...], [label, ...]
        '''
        true_pos = get_image_detector_mturk_images(self._det.id, True)
        false_pos = get_image_detector_mturk_images(self._det.id, False)
        return true_pos + false_pos, [POS_LABEL]*len(true_pos) + [NEG_LABEL]*len(false_pos)

    def get_old_classifier(self):
        ''' Reconstruct a classifier from the old detector.

        Returns:
            Spatial scene classifier.
        '''
        self._det.grab_files()
        return SpatialSceneClassifier.load_from_dir(self._det.local_dir())
