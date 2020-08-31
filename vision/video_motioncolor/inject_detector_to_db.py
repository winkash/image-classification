from logging import getLogger
from affine.model import VideoMotionColorDetector, Label, session
from affine.model.training_data import TrainingVideo
from affine.detection.utils import AbstractInjector
from affine.detection.vision.video_motioncolor.video_motioncolor_classifier\
    import VideoMotionColorClassifier

__all__ = ['VideoMotionColorDetectorInjector']
logger = getLogger(__name__)


class VideoMotionColorDetectorInjector(AbstractInjector):

    def get_names(self):
        """
            Returns the list of the file names required in the model
            folder of a Video Motion&Color based Classifier
        """
        return [VideoMotionColorClassifier.CONFIG_FILE,
                VideoMotionColorClassifier.COLOR_FILE,
                VideoMotionColorClassifier.MOTION_FILE]

    def inject_detector(
        self, detector_name, label_id, true_vid_list,
        confidence_th=None, acceptance_th=None
    ):
        """
        Create the detector in the table VideoMotionColorDetector, upload the tar
        file with all the model files to s3, and save the video_ids used as
        positive training data

        Args:
            detector_name: string with detector name
                (it has to be unique because this detector can't be replaced)
            label_id: int, target label id of the detector
            true_vid_list: list of ints that correspond to the ids of the videos
                used as positive training data
        Returns:
            det: the created VideoMotionColorDetector object

        Raise/Assertions:
            This function asserts if the label_id does not correspond to any
            existing label and if the detector_name already exists in the db
        """
        l = Label.get(label_id)
        assert (l != None), 'Label id %d does not correspond to any Label!'\
            % (label_id)

        det = VideoMotionColorDetector.by_name(detector_name)
        assert not det, 'VideoMotionColorDetector with name %s already exists!'\
            % detector_name

        det = VideoMotionColorDetector(name=detector_name)
        if confidence_th and acceptance_th:
            det.confidence_th = confidence_th
            det.acceptance_th = acceptance_th
        session.flush()
        det.add_targets([l])
        self.tar_and_upload(det)

        logger.info('VideoMotionColorDetector detector injected %s' % det)
        save_training_videos(det.id, true_vid_list)

        return det


def save_training_videos(detector_id, true_vid_list):
    """
    Store all the pairs (detector_id, video_id) in TrainingVideo table

    Args:
        detector_id
        true_vid_list: list of int values that correspond to positive training
            video ids

    NOTE: This function assumes the detector_id and video_ids exists, so should
    be checked before calling to inject_detector
    """
    true_vid_list = set(true_vid_list)
    for v_id in true_vid_list:
        TrainingVideo(detector_id=detector_id, video_id=v_id)
    session.flush()
