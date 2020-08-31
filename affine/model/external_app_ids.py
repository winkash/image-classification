from affine.model.base import *
from affine.model._sqla_imports import *

__all__ = ['ExternalAppId']

class ExternalAppId(Base):
    __tablename__ = 'external_app_ids'
    id = Column(Integer, primary_key=True)
    app_id = Column(Integer, ForeignKey('apps.id'), nullable=False)
    app_name = Column(UnicodeText, nullable=False)

    app = relation('App', backref=backref('external_app_ids', cascade='all')) 

    def __unicode__(self):
        return u'<ExternalAppId(%s)>' % self.app_name
