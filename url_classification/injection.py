from datetime import datetime
from logging import getLogger
import argparse
import sys

from affine.model import Label, UrlClassifier, session
from affine.model.classifier_models import UrlModel
from affine.detection.url_classification.url_config import MODEL_FILE,\
    CFG_FILE, CFG_SPEC
from affine.detection.utils import AbstractInjector
from affine.detection.utils.validate_config import validate_config

logger = getLogger(__name__)


class UrlModelInjector(AbstractInjector):

    def get_names(self):
        return [MODEL_FILE, CFG_FILE]

    def inject_model(self):
        cfg_obj = validate_config(self.model_path(CFG_FILE),
                                  CFG_SPEC)
        model_name = cfg_obj['model_name']
        label = Label.by_name(cfg_obj['target_label_name'])
        assert label
        url_model = UrlModel.create(name=model_name)
        self.tar_and_upload(url_model)
        logger.info('URL model injected %s' % url_model)


class UrlClassifierInjector(AbstractInjector):

    def get_names(self):
        return [CFG_FILE]

    def inject_classifier(self, replace_old):
        # TODO: This seems like it could be generalized for all classifiers
        cfg_obj = validate_config(self.model_path(CFG_FILE),
                                  CFG_SPEC)
        clf_name = cfg_obj['classifier_name']
        label = Label.by_name(cfg_obj['target_label_name'])
        assert label
        clf = UrlClassifier.by_name(clf_name)
        if replace_old:
            assert clf, 'UrlClassifier with name %s does not exist!'\
                % clf_name
        else:
            assert not clf, 'UrlClassifier with name %s already exists!'\
                % clf_name
            # create the new classifier
            clf = UrlClassifier.create(name=clf_name)
        # note that failures above while running the script does not roll back
        # previously inserted models
        self.tar_and_upload(clf)
        clf.updated_at = datetime.utcnow()
        session.flush()
        clf.add_targets([label])
        logger.info('URL classifier injected %s' % clf)


def main(cmd_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir',
                        help='Directory containing trained model')
    parser.add_argument('replace_old', type=int, choices=[0, 1],
                        help=('1 - Replace old classifier,'
                              ' 0 - Create a new classifier\n'
                              'classifier_name field in the config_file is'
                              'used to identify the classifier'))
    args = parser.parse_args(args=cmd_args)
    injector = UrlModelInjector(args.model_dir)
    injector.inject_model()
    injector = UrlClassifierInjector(args.model_dir)
    injector.inject_classifier(args.replace_old)


if __name__ == '__main__':
    main(sys.argv[1:])
