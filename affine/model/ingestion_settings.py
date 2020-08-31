"""Simple name/value store for ingestion settings"""
from datetime import datetime
from logging import getLogger

from affine.model.base import *
from affine.model._sqla_imports import *

__all__ = ['IngestionSettings']

logger = getLogger(__name__)


class IngestionSettings(Base):
    __tablename__ = "ingestion_settings"
    name = Column(VARCHAR(100), primary_key=True, nullable=False)
    value = Column(VARCHAR(1000), nullable=False)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __unicode__(self):
        return u'<%s: %s>' % (self.name, self.value)

    @classmethod
    def get_setting(cls, name, default_value):
        with session.begin():
            query = session.query(IngestionSettings.value).filter_by(name=name)
            value = query.first()
            if not value:
                return default_value
            return value[0]
