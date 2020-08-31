from logging import getLogger
from affine.detection.xmlrpc_server import XmlRpcServer, XmlRpcRequestHandler
from affine.model.classifier_models import UrlModel

logger = getLogger(__name__)

__all__ = ['UrlRequestHandler', 'UrlServer']


class UrlRequestHandler(XmlRpcRequestHandler):

    def __init__(self):
        self.classifiers = {}
        for url_model in UrlModel.query:
            self.classifiers[url_model.id] = url_model.get_data_processor()
        super(UrlRequestHandler, self).__init__()

    def list_models(self):
        """Return a list of models supported by this handler."""
        return self.classifiers.keys()

    def predict(self, model_id, url):
        """
        This function calls the underlying predict function from the data
        processor

        Args:
            model_id: int corresponding to the row in ClassifierModel
            url: url string

        Return:
            Prediction result from running model on url
        """
        return self.classifiers[model_id].predict(url)

    def predict_confidences(self, model_id, url):
        """
        This function calls the underlying predict function from the data
        processor

        Args:
            model_id: int corresponding to the row in ClassifierModel
            url: url string

        Return:
            Prediction confidence from running model on url
        """
        return self.classifiers[model_id].predict_confidences(url)


class UrlServer(XmlRpcServer):

    def setup(self):
        logger.info("Setting up URL request handler")
        self.url_request_handler = UrlRequestHandler()
        self.register_instance(self.url_request_handler)
