import xmlrpclib
from logging import getLogger

from affine import config
from affine.detection.xmlrpc_client import XmlRpcClient, retry_on_failure

logger = getLogger(__name__)

__all__ = ['LogoClient']

def file_to_binary(func):
    def wrapper(self, file_names):
        bin_data = []
        for file_name in file_names:
            with open(file_name, 'rb') as handle:
                bin_data.append(xmlrpclib.Binary(handle.read()))
        return func(self, bin_data)
    return wrapper


class LogoClient(XmlRpcClient):
    """This defines a client for Server."""

    def __init__(self, model_name, validate=False):
        host, port = (config.get('logo_server.host'),
                      config.get('logo_server.port'))
        super(LogoClient, self).__init__(host, port)
        if validate:
            error_msg = "{} not in list of available models".format(model_name)
            assert model_name in self.list_models(), error_msg
        self.model_name = model_name

    @file_to_binary
    @retry_on_failure
    def predict(self, bin_data):
        return self.server.predict(self.model_name, bin_data)

    @retry_on_failure
    def list_models(self):
        return self.server.list_models()
