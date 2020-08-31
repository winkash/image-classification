import logging
from affine.model._sqla_imports import Column, Integer, \
    String, UniqueConstraint, Unicode
from affine.model.detection import ModelGrabberMixin, Base
from affine.detection.cnn.caffe_processor import CaffeProcessor
from affine.detection.vision.logo_recognition.processor import LogoProcessor
from affine.detection.vision.vision_text_recognize.word_rec_processor import \
    WordRecProcessor
from affine.detection.url_classification.url_processor import UrlProcessor
from affine.detection.model_worker.config import DEFAULT_QUEUE
from affine.detection.vision.facerec.face_processor import FaceProcessor

logger = logging.getLogger(__name__)


class ClassifierModel(ModelGrabberMixin, Base):

    """
    Base class from which different models for classifiers will inherit

    This class expects that the inheriting class has a data_processor_cls
    attribute which in turn has a load_model method.

    This class's sole responsibility is to be able to provide an instance of
    the data_processor.
    """

    __tablename__ = "classifier_models"
    id = Column(Integer, primary_key=True)
    name = Column(Unicode(128), nullable=False)
    _cls = Column('cls', String(50), nullable=False)
    __mapper_args__ = dict(polymorphic_on=_cls)
    __table_args__ = (UniqueConstraint('cls', 'name',
                                       name='classifier_models_cls_name'),
                      {})
    _celery_queue = DEFAULT_QUEUE

    @property
    def celery_queue(self):
        return self._celery_queue

    @property
    def data_processor_cls(self):
        raise NotImplementedError

    def get_data_processor(self):
        self.grab_files()
        return self.data_processor_cls.load_model(self.local_dir())

    @property
    def tarball_prefix(self):
        raise NotImplementedError

    @property
    def tarball_basename(self):
        return "{}_{}_{}".format(self.tarball_prefix, self.name, self.id)


class CaffeModel(ClassifierModel):

    __mapper_args__ = dict(polymorphic_identity='caffe_models')
    data_processor_cls = CaffeProcessor
    tarball_prefix = 'caffe'
    _celery_queue = 'gpu'

    @property
    def celery_queue(self):
        """ Use a different queue for each CaffeModel """
        return self._celery_queue + '_' + self.name


class LogoRecModel(ClassifierModel):

    __mapper_args__ = dict(polymorphic_identity='logo_rec_models')
    data_processor_cls = LogoProcessor
    tarball_prefix = 'logo_rec'


class WordRecModel(CaffeModel):

    __mapper_args__ = dict(polymorphic_identity='wordrec_caffe_models')
    data_processor_cls = WordRecProcessor


class UrlModel(ClassifierModel):

    __mapper_args__ = dict(polymorphic_identity='url_models')
    data_processor_cls = UrlProcessor
    tarball_prefix = 'url'


class FaceModel(ClassifierModel):

    __mapper_args__ = dict(polymorphic_identity='face_models')
    data_processor_cls = FaceProcessor
    tarball_prefix = 'face'
    _celery_queue = 'faces'
