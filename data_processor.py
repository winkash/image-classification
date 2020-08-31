import tempfile
import os
from abc import abstractmethod, ABCMeta


def write_protect(func):
    def wrapper(self, *args, **kwargs):
        with self.lock:
            ret = func(self, *args, **kwargs)
        return ret
    return wrapper

def get_files(binaries):
    """Convert binaries to files.

    This function take a list of binaries and writes them to unique files
    under /tmp and returns the list of file names (absolute paths)

    Args:
        binaries: list of binaries

    Returns:
        fnames: list of absolute paths of the binaries written to files
    """
    fnames = []
    for image_data in binaries:
        tfd, tfname = tempfile.mkstemp()
        os.close(tfd)
        with open(tfname, 'wb') as tfh:
            tfh.write(image_data)
        fnames.append(tfname)
    return fnames


def file_from_binary(func):
    def wrapper(self, bin_data, *args, **kwargs):
        file_names = None
        try:
            assert (type(bin_data) == list), \
                    "Input should be a list"
            file_names = get_files(bin_data)
            return func(self, file_names, *args, **kwargs)
        finally:
            if file_names is not None:
                for file_name in file_names:
                    os.remove(file_name)
    return wrapper


class DataProcessor(object):
    __metaclass__ = ABCMeta

    # Python 3 has an abstractclassmethod which makes this better.
    @classmethod
    @abstractmethod
    def load_model(cls, *args, **kwargs):
        """
        This method should ideally take a model_dir and return an instance of
        the DataProcessor class
        """

    def poll(self):
        return True
