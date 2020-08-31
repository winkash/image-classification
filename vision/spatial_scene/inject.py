from affine.model import SpatialSceneDetector, SpatialSceneBetaDetector, session
from affine.detection.utils import AbstractInjector
from ..utils.scene_functions import save_training_images
from .spatial_scene_classifier import SpatialSceneClassifier

__all__ = ['SpatialSceneDetectorInjector', 'SpatialSceneBetaDetectorInjector']


class SpatialSceneDetectorInjector(AbstractInjector):
    detector_cls = SpatialSceneDetector

    def get_names(self):
        return SpatialSceneClassifier.model_names

    def inject(self, det_name, target_label, video_threshold,
               training_images, image_threshold=None):
        '''
        Inject model to db.

        If image_threshold is not specified, it will be set to that
        of the underlying classifier.

        Params:
            det_name: Name of detector created.
            target_label: Detector's target label.
            video_threshold: Detector's video threshold.
            training_images: ([(video_id, timestamp), ...], [label, ...])
            image_threshold: (Optional) Detector's image threshold.

        Returns:
            Detector.
        '''
        if image_threshold is None:
            clf = SpatialSceneClassifier.load_from_dir(self.model_dir)
            image_threshold = clf.image_threshold
        det = self.detector_cls(name=det_name, video_threshold=video_threshold,
                                image_threshold=image_threshold,
                                target_label=target_label)
        session.flush()
        save_training_images(det.id, *training_images)
        self.tar_and_upload(det)
        return det

class SpatialSceneBetaDetectorInjector(SpatialSceneDetectorInjector):
    detector_cls = SpatialSceneBetaDetector
