import SocketServer

from logging import getLogger
from SimpleXMLRPCServer import SimpleXMLRPCServer

__all__ = ['XmlRpcServer']

logger = getLogger(__name__)


class XmlRpcServer(SocketServer.ThreadingMixIn, SimpleXMLRPCServer):
    """Super class for all XmlRpcRequestHandlers.

    Inherit from this class in order to add functionality to request handlers
    as required.
    """
    allow_reuse_address = True

    def setup(self):
        self.register_introspection_functions()

    def stop_server(self):
        """Close connections and shutdown the server"""
        logger.info("Stopping server")
        self.shutdown()
        logger.info("Stopped server")


class XmlRpcRequestHandler(object):
    """Super class for all XmlRpcRequestHandlers.

    Inherit from this class in order to add functionality to request handlers
    as required.
    """

    def poll(self):
        return True
