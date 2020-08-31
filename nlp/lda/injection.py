import argparse
import os

from datetime import datetime
from logging import getLogger

from affine.model import Label, LdaDetector, session, LdaModel
from affine.detection.nlp.topic_model import PipelineRunner
from affine.detection.utils import AbstractInjector

logger = getLogger(__name__)

__all__ = ['LdaDetectorInjector', 'LdaModelInjector']


class LdaModelInjector(AbstractInjector):
    INFERENCER_FILENAME = 'inferencer_file'
    PIPE_FILENAME = 'pipe_file'
    TOPICS_FILENAME = 'topics'

    def get_names(self):
        return [self.INFERENCER_FILENAME,
                self.PIPE_FILENAME,
                self.TOPICS_FILENAME]

    def inject_model(self):
        lda_model = LdaModel()
        session.flush()
        self.tar_and_upload(lda_model)
        return lda_model


class LdaDetectorInjector(AbstractInjector):

    def get_names(self):
        self.config_obj = PipelineRunner.validate_config_file(
            self.model_path(PipelineRunner.CFG_NAME))

        # For LDA detectors
        return [PipelineRunner.CFG_NAME,
                self.config_obj['vocab_file'],
                self.config_obj['classifier_params']['model_file']]

    def inject_detector(self, lda_model_id, replace_old):
        target_label = Label.by_name(self.config_obj['target_label_name'])
        assert target_label, 'Label for %s does not exist!' % self.config_obj[
            'target_label_name']
        assert LdaModel.get(lda_model_id), "Invalid lda model id!"
        detector_name = self.config_obj['detector_name']
        det = LdaDetector.query.filter_by(name=detector_name).first()
        if replace_old:
            assert det, 'LdaDetector with name %s does not exist!' \
                    % detector_name
            det.lda_model_id = lda_model_id
        else:
            assert not det, 'LdaDetector with name %s already exists!' \
                    % detector_name
            # create the new detector
            det = LdaDetector(name=detector_name, lda_model_id=lda_model_id)

        session.flush()
        det.add_targets([target_label])

        self.tar_and_upload(det)
        det.updated_at = datetime.utcnow()
        session.flush()
        return det


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model-dir', dest='model_dir', required=True,
                        help='Directory that contains the model files')
    parser.add_argument('-replace-old', dest='replace_old', required=True,
                        type=int, choices=[0, 1],
                        help=('1 - Replace old model for existing detector, 0 '
                        '- Create a new detector\ndetector_name field in the '
                        'config_file is used to identify the detector'))
    parser.add_argument('-lda-model-id', dest='lda_model_id', required=True,
                        type=int, help='lda model id for the detector')
    args = parser.parse_args()
    assert os.path.exists(args.model_dir), 'model dir does not exist!'

    di = LdaDetectorInjector(args.model_dir)
    di.inject_detector(args.lda_model_id, args.replace_old)

if __name__ == '__main__':
    main()
