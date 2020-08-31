from logging import getLogger
from affine.detection.utils import AbstractInjector
from affine.detection.vision.vision_text_recognize.vision_text_recognizer \
    import VisionTextRecognizer
from affine.model import TextRecognizeClassifier, TextRecognizeBetaClassifier,\
    Label, session

__all__ = ['TextRecognizeClassifierInjector',
           'TextRecognizeBetaClassifierInjector']
logger = getLogger(__name__)


class TextRecognizeClassifierInjector(AbstractInjector):

    """ Injector class for a TextRecognizeClassifier """

    classifier_cls = TextRecognizeClassifier

    def get_names(self):
        """
            Returns the list of the file names required in the model
            folder of a VisionTextRecognizer
        """
        return [VisionTextRecognizer.CONFIG_FILE,
                VisionTextRecognizer.LEXICON]

    def inject_detector(self, detector_name, label_id, pred_thresh=None):
        """
        Create the detector in the table TextRecognizeClassifier, upload the
        tar file with all the model files to s3

        Args:
            detector_name: string with detector name
                (it has to be unique because this detector can't be replaced)
            label_id: int, target label id of the detector
            pred_thresh: set float threshold for word recognition

        Returns:
            det: the created TextRecognizeClassifier object

        Raise/Assertions:
            This function asserts if the label_id does not correspond to any
            existing label and if the detector_name already exists in the db
        """
        l = Label.get(label_id)
        assert l, 'Label id %d does not correspond to any Label!'\
            % (label_id)

        det = self.classifier_cls.by_name(detector_name)
        assert not det, '%s with name %s already exists!'\
            % (self.classifier_cls.__name__, detector_name)
        det = self.classifier_cls(name=detector_name)
        session.flush()
        det.add_targets([l])
        self.tar_and_upload(det)
        logger.info('%s detector injected %s'
                    % (self.classifier_cls.__name__, det))
        return det


class TextRecognizeBetaClassifierInjector(TextRecognizeClassifierInjector):

    """ Injector class for a TextRecognizeBetaClassifier """
    classifier_cls = TextRecognizeBetaClassifier
