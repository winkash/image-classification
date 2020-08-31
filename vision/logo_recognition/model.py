import os
import ast
import pickle

from .training import CFG_SPEC
from affine.detection.model.features import BagOfWords
from affine.detection.model.classifiers import KNNScikit
from affine.detection.vision.utils.scene_functions import get_config

__all__ = ['LogoModel']

class LogoModel(object):

    files = ['model.cfg', 'bow', 'knn', 'logos']

    def __init__(self, model_dir):
        """The LogoMatcher is used to encapsulate all the information of
        a trained logo-model, the model_dir needs the following files,
        bow, knn, logos, model.cfg
        """
        self.model_dir = model_dir
        self.bow = BagOfWords.load_from_dir(self.model_path('bow'))
        self.knn = KNNScikit.load_from_file(self.model_path('knn'))
        with open(self.model_path('logos')) as f:
            self.training_logos = pickle.load(f)
        self.cfg = get_config(self.model_path('model.cfg'), CFG_SPEC.split('\n'))

        # some params for performing matching
        self.k_neighbors = self.cfg['KNN']['k_neighbors']
        self.min_points = self.cfg['RbM']['min_points']
        self.min_matches = self.cfg['RbM']['min_matches']
        self.ransac_th = self.cfg['RbM']['ransac_th']
        self.accept_th = self.cfg['RbM']['accept_th']
        self.ransac_algorithm = self.cfg['RbM']['ransac_algorithm']
        self.ransac_max_iter = self.cfg['RbM']['ransac_max_iter']
        self.ransac_prob = self.cfg['RbM']['ransac_prob']
        self.inlier_r = self.cfg['RbM']['inlier_r']
        # size parameters
        self.resize = self.cfg['size']['resize']
        self.standard_width = self.cfg['size']['standard_width']
        # box finder params
        self.patch_shapes = ast.literal_eval(self.cfg['BOF']['patch_shapes'])
        self.scales = ast.literal_eval(self.cfg['BOF']['scales'])
        self.step_size = self.cfg['BOF']['step_size']
        self.center_area_offset = self.cfg['BOF']['center_area_offset']
        self.corner_area_sz = ast.literal_eval(self.cfg['BOF']['corner_area_sz'])
        self.raise_on_size = self.cfg['BOF']['raise_on_size']
        self.contrast_thresh = self.cfg['BOF']['contrast_thresh']
        self.variance_thresh = self.cfg['BOF']['variance_thresh']

    def model_path(self, name):
        return os.path.join(self.model_dir, name)

    def get_image_descs_and_target_label_ids(self, logo_ids):
        image_descs = [self.training_logos[n].image_desc for n in logo_ids]
        target_label_ids = [self.training_logos[n].target_label_id for n in logo_ids]
        return image_descs, target_label_ids
