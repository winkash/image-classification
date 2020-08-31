import cPickle
import sys
from kombu.serialization import BytesIO, register
from logging import getLogger

logger = getLogger(__name__)
# 1 MB
SIZE_LIMIT = 1 * 1024 * 1024
DEFAULT_PICKLE_NAME = 'custom_pickle'


def register_pickle(size_limit=SIZE_LIMIT, pickle_name=None):
    """Custom pickle registration which limits the size of payload"""
    pickle_name = pickle_name or DEFAULT_PICKLE_NAME

    def pickle_loads(s):
        # used to support buffer objects
        return cPickle.load(BytesIO(s))

    def pickle_dumps(obj):
        val = cPickle.dumps(obj, protocol=2)
        size = sys.getsizeof(val)
        assert size < size_limit, "Payload size is {}: Allowed is {}".\
                        format(size, size_limit)
        return val

    register(pickle_name, pickle_dumps, pickle_loads,
             content_type='application/x-python-serialize',
             content_encoding='binary')
