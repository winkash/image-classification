import os
import tempfile

from logging import getLogger
from affine.detection.xmlrpc_server import XmlRpcServer, XmlRpcRequestHandler
from affine.model.classifier_models import LogoRecModel

logger = getLogger(__name__)

__all__ = ['LogoRequestHandler', 'LogoServer']

def file_from_binary(func):
    def wrapper(self, model_name, bin_data):
        file_names = None
        try:
            file_names = get_files(bin_data)
            return func(self, model_name, file_names)
        finally:
            if file_names:
                map(os.remove, file_names)

    return wrapper

def get_files(binaries):
    """Convert xml binaries to files.

    This function take a list of xml binaries and writes them to unique files
    under /tmp and returns the list of file names (absolute paths)

    Args:
        binaries: list of binaries of type xmlrpclib.Binary

    Returns:
        fnames: list of absolute paths of the binaries written to files
    """
    fnames = []
    for image_data in binaries:
        tfd, tfname = tempfile.mkstemp()
        os.close(tfd)
        with open(tfname, 'wb') as tfh:
            tfh.write(image_data.data)
        fnames.append(tfname)
    return fnames

class LogoRequestHandler(XmlRpcRequestHandler):

    def __init__(self):
        self._processors = {}
        for model in LogoRecModel.query:
            self._processors[model.name] = model.get_data_processor()
        super(LogoRequestHandler, self).__init__()

    def list_models(self):
        """Return a list of models supported by this handler."""
        return self._processors.keys()

    @file_from_binary
    def predict(self, model_name, file_names):
        """
        This function calls the underlying predict function from the data
        processor

        Args:
            model_name: LogoRecModel name
            file_names: list of input images

        Return:
            Prediction results on list of images
        """
        return self._processors[model_name].predict(file_names)


class LogoServer(XmlRpcServer):

    def setup(self):
        logger.info("Setting up Logo request handler")
        self.logo_request_handler = LogoRequestHandler()
        self.register_instance(self.logo_request_handler)
