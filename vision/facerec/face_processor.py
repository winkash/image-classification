import os
from tempfile import mkstemp

from affine.detection.data_processor import DataProcessor
from affine.detection.vision.utils.scene_functions import get_config
from .face_indexing import FaceIndex
from .face_sig_extract import FaceSigExtractor
from .injection import FACE_MODEL_INDEX_FILE, FACE_MODEL_CONFIG_FILE,\
    FACE_MODEL_CONFIG_SPEC

__all__ = ['FaceProcessor']

class FaceProcessor(DataProcessor):

    def __init__(self, model_dir):
        face_index_path = os.path.join(model_dir, FACE_MODEL_INDEX_FILE)
        config_path = os.path.join(model_dir, FACE_MODEL_CONFIG_FILE)
        self.face_index = FaceIndex.load_from_file(face_index_path)
        self.config = get_config(config_path, FACE_MODEL_CONFIG_SPEC.split('\n'))

    @classmethod
    def load_model(cls, model_dir):
        return cls(model_dir)

    def predict(self, bin_data, width, height):
        fd, image_path = mkstemp(suffix='.jpg')
        try:
            os.close(fd)
            self._reconstruct_image(bin_data, image_path)
            rect = (0, 0, width, height)
            fse = FaceSigExtractor()
            [signature], [(conf, parts)] = fse.extract([(image_path, rect)])
            neighbors = self.face_index.get_neighbors(signature)
            label_id, _ = self.face_index.get_verdict(
                    neighbors, radius=self.config['radius'],
                    prob_delta=self.config['prob_delta'], k=self.config['k'])
            return label_id, conf, parts
        finally:
            os.remove(image_path)

    @staticmethod
    def _reconstruct_image(bin_data, path):
        with open(path, 'wb') as handle:
            handle.write(bin_data)
