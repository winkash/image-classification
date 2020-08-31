import calendar
from datetime import datetime
import os
import tarfile
from tempfile import mkdtemp

from affine import config
from affine.aws import s3client
from affine.model.base import *
from affine.model._sqla_imports import *

__all__ = ['TextDetectionVersion']


class TextDetectionVersion(Base):
    """Table containing all versions for Abstract Text Detectors"""
    __tablename__ = "text_detection_versions"
    id = Column(Integer, primary_key=True)
    detector_type = Column(Enum('topic_model', strict = True), nullable=False)
    timestamp = Column(Timestamp, nullable=False)

    TM_MODEL_BASE_NAME = 'classifier_model_files'
    TM_CATEGORY_PICKLE_NAME = "category_info.pickle"
    TM_TOPIC_MAP_NAME = "youtube.topic_map"
    TM_PIPE_FILE_NAME = "youtube.pipe_file"

    def __unicode__(self):
        return u'<TextDetectionVersion(%s) type:%s timestamp:%s>' %(self.id, self.detector_type, self.timestamp)

    @classmethod
    def get_current_timestamp(cls):
        d = datetime.utcnow()
        return calendar.timegm(d.utctimetuple())

    @classmethod
    def get_latest_version_timestamp(cls, detector_type=None, return_int=False):
        # now only supports old topic model detectors
        query = session.query(func.max(cls.timestamp))
        if detector_type is None:
            ts = query.scalar()
        elif detector_type == 'topic_model':
            ts = query.filter_by(detector_type=detector_type).scalar()
        else:
            raise ValueError('Invalid detector_type %s' %detector_type)
        if ts and return_int:
            ts = calendar.timegm(ts.utctimetuple())
        return ts

    @classmethod
    def create_versioned_tarball(cls, classifier_model_files_dir, op_dir, timestamp_int):
        """ Create tarball with correct version naming from folder with model files"""
        model_files = os.listdir(classifier_model_files_dir)
        assert cls.TM_CATEGORY_PICKLE_NAME in model_files, "Category pickle file not found!"
        assert cls.TM_TOPIC_MAP_NAME in model_files, "Tier 1 topic map file not found!"
        assert cls.TM_PIPE_FILE_NAME in model_files, "Pipe file not found!"
        versioned_classifier_model_files_name = "%s_%d" %(cls.TM_MODEL_BASE_NAME, timestamp_int)
        versioned_tarball_name = "%s.tar.gz" %versioned_classifier_model_files_name
        versioned_tarball_path = os.path.join(op_dir, versioned_tarball_name)
        with tarfile.open(versioned_tarball_path, 'w:gz') as tf:
            tf.add(classifier_model_files_dir, arcname=versioned_classifier_model_files_name)
        return (versioned_tarball_path, versioned_tarball_name)

    @classmethod
    def create_new_tm_version(cls, classifier_model_files_dir, op_dir=None):
        """Creates new versioned tarball and uploads it to the s3 affine detector bucket"""
        assert os.path.isdir(classifier_model_files_dir), "Directory missing : %s"%classifier_model_files_dir
        if op_dir is None:
            op_dir = mkdtemp()
        if not os.path.isdir(op_dir):
            os.makedirs(op_dir)
        print 'Storing model files at location: %s' %op_dir
        # get new version and folder with appropriate name
        timestamp_int = cls.get_current_timestamp()
        # create tarball with version from input folder 
        versioned_tarball_path, versioned_tarball_name = cls.create_versioned_tarball(classifier_model_files_dir, op_dir, timestamp_int)
        # upload to s3
        bucket = config.s3_detector_bucket()
        s3client.upload_to_s3(bucket, versioned_tarball_name, versioned_tarball_path, public=False)
        # create new row in DB
        tmv = cls(detector_type='topic_model', timestamp=datetime.utcfromtimestamp(timestamp_int))
        session.flush()
        return tmv
