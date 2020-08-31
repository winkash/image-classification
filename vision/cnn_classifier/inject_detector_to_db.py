from logging import getLogger

from affine.model import CnnClassifier, CnnBetaClassifier, Label, session
from affine.detection.utils import AbstractInjector
from affine.detection.vision.cnn_classifier.cnn_classifier_flow_factory \
    import CnnClassifierFlowFactory
from affine.detection.vision.video_motioncolor.inject_detector_to_db \
    import save_training_videos

__all__ = ['CnnClassifierInjector']
logger = getLogger(__name__)


class CnnClassifierInjector(AbstractInjector):
    detector_cls = CnnClassifier

    def get_names(self):
        """
            Returns the list of the file names required in the model
            folder of a Video Motion&Color based Classifier
        """
        return [CnnClassifierFlowFactory.CONFIG_FILE]

    def get_optional_file_names(self):
        classifier_file_dict = CnnClassifierFlowFactory.CLASSIFIER_FILE
        return [pickle_file for pickle_file in classifier_file_dict.values()]

    def inject_detector(
        self, detector_name, list_label_ids, true_vid_list=None
        ):
        """
        Create the detector in the table Cnn classifier, upload the tar
        file with all the model files to s3, and save the video_ids used as
        positive training data

        Args:
            detector_name: string with detector name
                (it has to be unique because this detector can't be replaced)
            label_id: int, target label id of the detector
            true_vid_list: list of ints with ids of the videos used
                as positive training data
        Returns:
            det: the created CnnClassifier object

        Raise/Assertions:
            This function asserts if the label_id does not correspond to any
            existing label and if the detector_name already exists in the db
        """
        target_label_list = {Label.get(label_id) for label_id in list_label_ids
                             if Label.get(label_id)}
        target_label_list = list(target_label_list)
        assert len(target_label_list),\
            "Target label list needs at least one Label that exists in the DB"

        det = self.detector_cls.by_name(detector_name)
        assert not det, 'Cnn Classifier with name %s already exists!'\
            % detector_name

        det = self.detector_cls(name=detector_name)
        session.flush()
        det.add_targets(target_label_list)
        self.tar_and_upload(det)
        logger.info('CnnClassifier detector injected %s' % det)

        if true_vid_list:
            save_training_videos(det.id, true_vid_list)

        return det


class CnnBetaClassifierInjector(CnnClassifierInjector):

    """
    Injector class for the Beta version of CnnClassifier

    It extends CnnClassifierInjector with the same behaviour, except that the
    classifier is injected in CnnBetaClassifier table.

    """

    detector_cls = CnnBetaClassifier
