import os
import tarfile
from tempfile import mkdtemp

from affine import config
from affine.aws import s3client

__all__ = ['AbstractInjector']


class AbstractInjector(object):
    """ Abstract detector injection base class inherited by
    various detector injection classes.

    Args:
        model_dir : directory containing all required model files/directories
        has_mat_file: bool flag for whether the model is to be loaded from a
        mat file.

    Returns:
        None

    For a specific example on inheriting this class, look for instance
    at affine/affine/detection/nlp/lda/injection.py
    """
    def __init__(self, model_dir, optional_files=None, has_mat_file=False):
        self.has_mat_file = has_mat_file
        self.model_dir = model_dir
        self.names = []
        if optional_files is not None:
            valid_optional_files = self.get_optional_file_names()
            for file_name in optional_files:
                assert file_name in valid_optional_files
            self.names.extend(optional_files)
        self.names.extend(self.get_names())
        self.check_model_paths()

    @property
    def model_paths(self):
        """ List of full paths to the model files/directories """
        return [self.model_path(name) for name in self.names]

    def get_names(self):
        """ The inheriting class must implement this method to return
        the list of file or directory basenames expected to be in the model tarball. For
        example if a model requires file 'foo' and directory 'bar' to be in the
        model tarball, this function should return ['foo', 'bar']
        """
        raise NotImplementedError

    def model_path(self, name):
        return os.path.join(self.model_dir, name)

    def check_model_paths(self):
        """ Check that model files/directories actually exist in the model directory """
        for path in self.model_paths:
            assert os.path.exists(path), 'Path %s does not exist' % path

    def get_optional_file_names(self):
        """ The inheriting class must implement this method to return
        the list of file basenames optional in the model tarball. For
        example if a model requires the files 'foo' and 'bar' to be in the
        model tarball, this function should return ['foo', 'bar']
        """
        return []

    @classmethod
    def create_tarball(cls, model_paths, model_dir_name, op_dir=None):
        """Generic method to create roll up model files/directories into a tarball

        Args:
            model_paths: List of paths to model files/directories
            model_dir_name: Top level directory to put all model files/directories under
                for the tarball. The tarball's name is the same with .tar.gz
                extension
            op_dir: Directory to write the tarball to. If unspecified, the
            tarball is written to a temporary directory

        Returns:
            Model tarball
        """
        if op_dir is None:
            op_dir = mkdtemp()
        assert not model_dir_name.endswith('.tar.gz')
        tarball_path = os.path.join(op_dir, model_dir_name + '.tar.gz')
        with tarfile.open(tarball_path, "w:gz") as tar:
            for path in model_paths:
                name = os.path.basename(os.path.normpath(path))
                tar.add(path, arcname=os.path.join(model_dir_name, name))
        return tarball_path

    def tar_and_upload(self, clf):
        """ Creates tarball with model files/directories and upload to s3

        Args:
            clf: classifier or model that has a tarball_basename property
                The uploaded tarball will have this name with a .tar.gz
                extension.
        """
        tarball_path = self.create_tarball(self.model_paths, clf.tarball_basename,
                                           op_dir=self.model_dir)
        bucket = config.s3_detector_bucket()
        s3client.upload_to_s3(bucket, os.path.basename(tarball_path),
                              tarball_path, public=False)
