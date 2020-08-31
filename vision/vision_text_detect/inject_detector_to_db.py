from logging import getLogger
from affine.detection.utils import AbstractInjector
from affine.detection.vision.vision_text_detect.vision_text_detector \
    import VisionTextDetector
from affine.model import TextDetectClassifier, TextDetectBetaClassifier, \
    Label, session

__all__ = ['TextDetectClassifierInjector', 'TextDetectBetaClassifierInjector']
logger = getLogger(__name__)


class TextDetectClassifierInjector(AbstractInjector):

    """ Injector class for a TextDetectClassifier """

    classifier_cls = TextDetectClassifier

    def get_names(self):
        """Returns the list of the file names required in the model
            folder of a VisionTextDetector
        """
        return [VisionTextDetector.CONFIG_FILE,
                VisionTextDetector.WORD_DET_RFC,
                VisionTextDetector.REGRESSION_PARAMS]

    def inject_detector(self, detector_name, label_id, pred_thresh=None):
        """Create the detector in the table TextDetectClassifier, upload the tar
        file with all the model files to s3

        Args:
            detector_name: string with detector name
                (it has to be unique because this detector can't be replaced)
            label_id: int, target label id of the detector
            pred_thresh: set float threshold for word detection

        Returns:
            clf: the created TextDetectClassifier object

        Raise/Assertions:
            This function asserts if the label_id does not correspond to any
            existing label and if the detector_name already exists in the db
        """
        l = Label.get(label_id)
        assert l, 'Label id %d does not correspond to any Label!' % (label_id)
        clf = self.classifier_cls.by_name(detector_name)
        assert not clf, '%s with name %s already exists!'\
            % (self.classifier_cls.__name__, detector_name)

        clf = self.classifier_cls(name=detector_name,
                                  pred_thresh=pred_thresh)
        session.flush()
        clf.add_targets([l])
        self.tar_and_upload(clf)
        logger.info('%s detector injected %s'
                    % (self.classifier_cls.__name__, clf))
        return clf


class TextDetectBetaClassifierInjector(TextDetectClassifierInjector):

    """ Injector class for a TextDetectBetaClassifier """

    classifier_cls = TextDetectBetaClassifier
