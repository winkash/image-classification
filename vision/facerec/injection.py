import os
from tempfile import mkdtemp

from affine.detection.utils import AbstractInjector
from affine.detection.vision.utils.scene_functions import get_config
from affine.model import session, FaceRecognizeClassifier, Label, Box, TrainingBox


__all__ = ['FaceModelInjector', 'FaceRecognizeClassifierInjector']


FACE_MODEL_INDEX_FILE = 'face_index.pickle'
FACE_MODEL_CONFIG_FILE = 'config.cfg'

FACE_MODEL_CONFIG_SPEC =\
"""
radius = float
prob_delta = float
k = integer
"""

FACE_REC_CONFIG_SPEC = "model_name = string"
FACE_REC_CONFIG_FILE = "config.cfg"

class FaceModelInjector(AbstractInjector):

    def get_names(self):
        return [FACE_MODEL_INDEX_FILE, FACE_MODEL_CONFIG_FILE]

    def inject(self, name):
        from affine.model.classifier_models import FaceModel
        model = FaceModel.create(name=name)
        self.tar_and_upload(model)
        return model


class FaceRecognizeClassifierInjector(AbstractInjector):

    def get_names(self):
        return [FACE_REC_CONFIG_FILE]

    def inject(self, clf_name, target_label_ids, training_box_ids):
        """'target_label_ids' and 'training_box_ids' should be sets"""
        target_labels = [Label.get(l_id) for l_id in target_label_ids]
        assert all(target_labels), "Bad target label id"
        training_boxes = [Box.get(b_id) for b_id in training_box_ids]
        assert all(training_boxes), "Bad training box id"
        clf = FaceRecognizeClassifier.create(name=clf_name)
        clf.add_targets(target_labels)
        for box_id in training_box_ids:
            TrainingBox(detector_id=clf.id, box_id=box_id)
        session.flush()
        self.tar_and_upload(clf)
        return clf

    @staticmethod
    def make_model_dir(model_name):
        """Build a model directory for FaceRecognizeClassifer.
            
        The directory contains only a config file listing 'model_name',
        the name of the FaceModel to use.
        """
        model_dir = mkdtemp()
        config_file = os.path.join(model_dir, FACE_REC_CONFIG_FILE)
        with open(config_file, 'w') as f:
            f.write('model_name = {}'.format(model_name))
        return model_dir

    @staticmethod
    def get_model_name(model_dir):
        """Extract the FaceModel name from a FaceRecognizeClassifer's model directory."""
        model_file = os.path.join(model_dir, FACE_REC_CONFIG_FILE)
        cfg = get_config(model_file, FACE_REC_CONFIG_SPEC.split('\n'))
        return cfg['model_name']
