import xmlrpclib
from logging import getLogger
from affine.retries import retry_operation

logger = getLogger(__name__)

__all__ = ['XmlRpcClient']

def retry_on_failure(func):
    def wrapper(*args, **kwargs):
        kwargs['num_tries'] = 5
        kwargs['sleep_time'] = 1
        kwargs['error_class'] = Exception
        return retry_operation(func, *args, **kwargs)
    return wrapper


class XmlRpcClient(object):

    def __init__(self, host, port):
        # ServerProxy defines a default timeout of 10s.
        # This is only if the server has gone away. Hence, unless we
        # specifically want to change this, we can keep it the same. When we do
        # want to change it, the link below might be useful.

        # http://stackoverflow.com/questions/372365/
        # set-timeout-for-xmlrpclib-serverproxy

        self.server = xmlrpclib.ServerProxy("http://%s:%s" % (host, port))

    def poll(self):
        try:
            return retry_operation(self.server.poll, num_tries=5,
                                   sleep_time=1, error_class=Exception)
        except Exception:
            return False
