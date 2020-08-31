from affine.model._sqla_imports import *
from affine.model.base import Base

__all__ = [
    'TrainingImage',
    'TrainingPage',
    'TrainingBox',
    'TrainingVideo',
    'LabelTrainingPage']


class TrainingImage(Base):
    __tablename__ = 'training_images'

    id = Column(Integer, primary_key=True)
    detector_id = Column(
        Integer,
        ForeignKey('abstract_detectors.id'),
        nullable=False)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False)
    timestamp = Column(Integer, nullable=False)
    label = Column(Integer, nullable=False)


class TrainingPage(Base):
    __tablename__ = 'training_pages'

    id = Column(Integer, primary_key=True)
    detector_id = Column(
        Integer,
        ForeignKey('text_detectors.id'),
        nullable=False)
    page_id = Column(Integer, ForeignKey('web_pages.id'), nullable=False)
    detector_version = Column(DateTime, nullable=False)

    def __unicode__(self):
        return "<TrainingPage detector_id(%s) page_id(%s)>" % (
            self.detector_id, self.page_id)


class TrainingBox(Base):
    __tablename__ = 'training_boxes'

    detector_id = Column(
        Integer,
        ForeignKey('abstract_detectors.id'),
        primary_key=True)
    box_id = Column(Integer, ForeignKey('boxes.id'), primary_key=True)


class TrainingVideo(Base):
    __tablename__ = 'training_videos'

    id = Column(Integer, primary_key=True)
    detector_id = Column(
        Integer,
        ForeignKey('abstract_detectors.id'),
        nullable=False)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False)


class LabelTrainingPage(Base):
    __tablename__ = 'label_training_pages'

    label_id = Column(Integer, ForeignKey('labels.id'), primary_key=True)
    page_id = Column(Integer, ForeignKey('web_pages.id'), primary_key=True)

    def __unicode__(self):
        return "<LabelTrainingPage label_id(%s) page_id(%s)>" % (
            self.label_id, self.page_id)

    @classmethod
    def get_all_training_page_ids(cls, label_id):
        return [i.page_id for i in cls.query.filter_by(label_id=label_id)]
