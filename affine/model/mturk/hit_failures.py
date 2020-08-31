from datetime import datetime
import traceback

from affine.model._sqla_imports import *
from affine.model.base import *


__all__ = ['MTurkHitFailure']


class MTurkHitFailure(Base):
    __tablename__ = 'mturk_hit_failures'
    id = Column(Integer, primary_key=True)
    hit_id = Column(VARCHAR(128), nullable=False)
    worker_id = Column(VARCHAR(128), nullable=True, default=None)
    message = Column(UnicodeText, nullable=False, default=traceback.format_exc)
    created = Column(DateTime, nullable=False, default=datetime.utcnow)
    deleted = Column(Boolean, nullable=False, default=False)
