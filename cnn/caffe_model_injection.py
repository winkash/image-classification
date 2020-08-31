import argparse
import os

from logging import getLogger

from affine.detection.utils import AbstractInjector
from affine.model.classifier_models import CaffeModel, WordRecModel
from affine.model import session
from affine.detection.cnn.caffe_processor import DEPLOY, CAFFE_MODEL, \
    MEAN_IMAGE, TRAIN_TEST, SOLVER, LABELS
from affine.detection.vision.vision_text_recognize.word_rec_processor import \
    MAT_FILE
logger = getLogger(__name__)

__all__ = ['CaffeModelInjector']


class CaffeModelInjector(AbstractInjector):

    def inject_model(self, name):
        if self.has_mat_file:
            caffe_model = WordRecModel(name=name)
        else:
            caffe_model = CaffeModel(name=name)
        session.flush()
        self.tar_and_upload(caffe_model)
        return caffe_model

    def get_names(self):
        if self.has_mat_file:
            return [DEPLOY, MAT_FILE]
        else:
            return [DEPLOY, CAFFE_MODEL]

    def get_optional_file_names(self):
        return [MEAN_IMAGE, TRAIN_TEST, SOLVER, LABELS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-dir', dest='model_dir', required=True,
                        help='Directory that contains the model files')
    parser.add_argument('-o', '--optional-files', type=str, nargs='+',
                        dest='optional_files', help='Optional files which can'
                        'be in the model')
    parser.add_argument('-n', '--name', type=str, required=True,
                        dest='name', help='Name of CaffeModel')
    parser.add_argument('-M', '--has-mat-file', type=bool, dest='has_mat_file',
                         help='Optional flag for loading model from mat file')
    args = parser.parse_args()
    assert os.path.exists(args.model_dir), 'model dir does not exist!'

    model_injector = CaffeModelInjector(args.model_dir,
            optional_files=args.optional_files, has_mat_file=args.has_mat_file)
    model_injector.inject_model(args.name)

if __name__ == '__main__':
    main()
