import shutil
import argparse
import os

from tempfile import mkdtemp
from logging import getLogger

from affine.detection.utils import AbstractInjector
from affine.model import Label, LogoClassifier, LogoBetaClassifier
from affine.model.classifier_models import LogoRecModel
from affine.detection.vision.logo_recognition.model import LogoModel

logger = getLogger(__name__)

__all__ = ['LogoClassifierInjector', 'LogoBetaClassifierInjector',
           'LogoRecModelInjector', 'inject_classifier_and_model']


CFG_SPEC = "model_name = string"
CFG_FILE_NAME = "model.cfg"


class LogoClassifierInjector(AbstractInjector):

    _clf_cls = LogoClassifier

    def get_names(self):
        return [CFG_FILE_NAME]

    def inject(self, clf_name, target_label_ids):
        '''
        Injects classifier and classifier targets to DB.

        Params:
            clf_name: Classifier's name.
            target_label_ids: The target label ids classifier's 'predict' gives

        Returns:
            Classifier.
        '''
        clf = self._clf_cls.create(name=clf_name)
        target_labels = [Label.get(l_id) for l_id in target_label_ids]
        assert all(target_labels), "Bad target label id"
        clf.add_targets(target_labels)
        self.tar_and_upload(clf)
        return clf


class LogoBetaClassifierInjector(LogoClassifierInjector):
    _clf_cls = LogoBetaClassifier


class LogoRecModelInjector(AbstractInjector):

    def inject(self, name):
        logo_model = LogoRecModel.create(name=name)
        self.tar_and_upload(logo_model)
        return logo_model

    def get_names(self):
        return LogoModel.files


def inject_classifier_and_model(model_dir, clf_name, model_name, beta=False):
    """Injects both logo classifier and model.

    Args:
        model_dir: Logo model directory
        clf_name: Name of classifier
        model_name: Name of model
        beta: Should the classifier be beta?

    Returns:
        (clf, model)
    """
    clf_inj_cls = LogoBetaClassifierInjector if beta else LogoClassifierInjector
    model_inj = LogoRecModelInjector(model_dir)
    logger.info('Injecting model %s' % model_name)
    model = model_inj.inject(model_name)
    logger.info('Successfully injected model %s' % model_name)
    clf_model_dir = _make_classifier_model_dir(model_name)
    try:
        clf_inj = clf_inj_cls(clf_model_dir)
        logo_model = LogoModel(model_dir)
        target_label_ids = {logo.target_label_id for logo in logo_model.training_logos}
        logger.info('Injecting classifier %s' % clf_name)
        clf = clf_inj.inject(clf_name, target_label_ids)
        logger.info('Successfully injected classifier %s' % clf_name)
        return clf, model
    finally:
        shutil.rmtree(clf_model_dir)


def _make_classifier_model_dir(model_name):
    model_dir = mkdtemp()
    model_file = os.path.join(model_dir, CFG_FILE_NAME)
    with open(model_file, 'w') as f:
        f.write('model_name = {}'.format(model_name))
    return model_dir


def _parse_args(cmdline_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Directory that contains the model files')
    parser.add_argument('clf_name', help='Name of classifier')
    parser.add_argument('model_name', help='Name of LogoRecModel')
    parser.add_argument('--beta', default=False, action='store_true',
                        help='Inject a beta classifier instead')
    return parser.parse_args(args=cmdline_args)


def main(cmdline_args=None):
    args = _parse_args(cmdline_args)
    assert os.path.exists(args.model_dir), 'model dir does not exist!'
    clf, model = inject_classifier_and_model(args.model_dir, args.clf_name,
                                             args.model_name, beta=args.beta)
    logger.info("Injected classifier = %s, model = %s" % (clf, model))
    return clf, model


if __name__ == '__main__':
    main()
