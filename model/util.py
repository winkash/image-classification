import pickle

__all__ = ['PickleMixin']


class PickleMixin(object):
    """Class to convert between pickle files and objects"""
    @classmethod
    def load_from_file(cls, file_name):
        """Laod from a pickle file"""
        with open(file_name, 'rb') as f:
            return pickle.load(f)

    def save_to_file(self, file_name):
        """Save to a pickle file"""
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
