import cPickle as pickle
import os
from logging import getLogger
from affine.detection.data_processor import DataProcessor
from affine.detection.url_classification.url_config import MODEL_FILE

logger = getLogger(__name__)


class UrlProcessor(DataProcessor):

    def __init__(self, tfv, clf):
        self.tfv = tfv
        self.clf = clf

    def predict(self, url):
        ftrs = self.tfv.transform([url])
        return int(self.clf.predict(ftrs)[0])

    def predict_confidences(self, url):
        # will fail if the classifier doesn't support probabilities
        ftrs = self.tfv.transform([url])
        return self.clf.predict_proba(ftrs)[0].tolist()

    @classmethod
    def load_model(cls, model_dir):
        """
        Method returns instance of class UrlProcessor for a valid model_id

        Args:
            model_id: primary key from table url_models

        Returns:
            UrlProcessor instance
        """
        model_file_path = os.path.join(model_dir, MODEL_FILE)
        logger.info('Loading url models')
        with open(model_file_path, 'rb') as fi:
            tfv, clf = pickle.load(fi)
        return cls(tfv, clf)
