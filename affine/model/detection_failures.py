from datetime import datetime
import socket
import traceback

from affine.model._sqla_imports import *
from affine.model.base import *
from affine.uuids import str_to_uuid


__all__ = ['VideoDetectionFailure', 'TextDetectionFailure', 'RecalcFailure']


class VideoDetectionFailure(Base):
    __tablename__ = 'video_detection_failures'
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False)
    hostname = Column(CHAR(255), nullable=False, default=socket.gethostname)
    message = Column(UnicodeText, nullable=False, default=traceback.format_exc)
    created = Column(DateTime, nullable=False, default=datetime.utcnow)

    video = relation('Video', backref=backref('detection_failures', passive_deletes=True))


class TextDetectionFailure(Base):
    __tablename__ = 'text_detection_failures'
    id = Column(Integer, primary_key=True)
    page_id = Column(Integer, ForeignKey('web_pages.id'), nullable=False)
    hostname = Column(CHAR(255), nullable=False, default=socket.gethostname)
    message = Column(UnicodeText, nullable=False, default=traceback.format_exc)
    created = Column(DateTime, nullable=False, default=datetime.utcnow)

    page = relation('WebPage', backref=backref('detection_failures', passive_deletes=True))


class RecalcFailure(Base):
    __tablename__ = 'recalc_failures'
    id = Column(Integer, primary_key=True)
    page_ids = Column(String, nullable=False)
    page_ids_uuid = Column(CHAR(36), nullable=False)
    
    hostname = Column(CHAR(255), nullable=False, default=socket.gethostname)
    message=Column(UnicodeText, nullable=False, default=traceback.format_exc)
    created = Column(DateTime, nullable=False, default=datetime.utcnow)

    def reset_page_ids_uuid(self):
        "return self for chaining purposes"
        self.page_ids_uuid = str_to_uuid(self.page_ids)
        return self
