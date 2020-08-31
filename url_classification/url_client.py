from logging import getLogger
from affine import config
from affine.detection.xmlrpc_client import XmlRpcClient, retry_on_failure

logger = getLogger(__name__)

__all__ = ['UrlClient']


class UrlClient(XmlRpcClient):
    """This defines a client for UrlServer.

    Use an instance of this class in order to get predictions/features for
    images from a server process hosted at the value defined in
    'url_server.host' on port 'url_server.port' in config.
    """

    def __init__(self, model_id, validate_model_id=True):
        """
        Establish a connection with the server.

        Args:
            model_id: id from ClassifierModel table.
        """
        host, port = (config.get('url_server.host'),
                      config.get('url_server.port'))
        super(UrlClient, self).__init__(host, port)
        self._model_id = model_id
        if validate_model_id:
            self._check_model_id(model_id)

    def _check_model_id(self, val):
        if val not in self.list_models():
            raise ValueError("{} not in list of available models".format(val))

    @retry_on_failure
    def predict(self, url):
        return self.server.predict(self._model_id, url)

    @retry_on_failure
    def predict_confidences(self, url):
        return self.server.predict_confidences(self._model_id, url)

    @retry_on_failure
    def list_models(self):
        return self.server.list_models()
