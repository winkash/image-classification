from datetime import datetime
from logging import getLogger
import argparse

from affine.model import NerDetector, session, Label
from affine.model.training_data import TrainingPage
from affine.detection.utils import AbstractInjector

logger = getLogger(__name__)


class NerDetectorInjector(AbstractInjector):

    def get_names(self):
        return [NerDetector.SVM_MODEL]

    def inject_detector(self, detector_name, label_id, replace_old,
                        true_pid_file):
        l = Label.get(label_id)
        assert l is not None, "Label with id %s does not exist" %label_id
        det = NerDetector.by_name(detector_name)
        if replace_old:
            assert det, 'NerDetector with name %s does not exist!'\
                % detector_name
        else:
            assert not det, 'NerDetector with name %s already exists!'\
                % detector_name
            # create the new detector
            det = NerDetector(name=detector_name)
            session.flush()
            det.add_targets([l])

        self.tar_and_upload(det)
        det.updated_at = datetime.utcnow()
        session.flush()
        logger.info('NER detector injected %s' % det)
        save_training_pages(det.id, det.updated_at, true_pid_file)


def save_training_pages(detector_id, updated_at, true_pid_file):
    true_pid_list = open(true_pid_file, 'r').read().splitlines()
    true_pid_list = set(map(int, true_pid_list))
    for pid in true_pid_list:
        TrainingPage(
            detector_id=detector_id, detector_version=updated_at, page_id=pid)
    session.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model-dir', dest='model_dir', required=True,
                        help='Directory containing trained model')
    parser.add_argument('-target-label-id', dest='label_id', required=True,
                        type=int, help='Target label id')
    parser.add_argument('-detector-name', dest='detector_name', required=True,
                        help='Name of detector')
    parser.add_argument('-replace-old', dest='replace_old', required=True,
                        type=int, choices=[0, 1],
                        help=('1 - Replace old model for existing detector,'
                              ' 0 - Create a new detector\n'
                              'detector_name field in the config_file is used'
                              'to identify the detector'))
    parser.add_argument('-training-page-ids', dest='true_pid_file',
                        required=True,
                        help=('File with one true page id per'
                              ' line used for training'))
    args = parser.parse_args()
    di = NerDetectorInjector(args.model_dir)
    di.inject_detector(args.detector_name, args.label_id,
                       args.replace_old, args.true_pid_file)


if __name__ == '__main__':
    main()
