from affine.model.base import *
from affine.model._sqla_imports import *
import json

__all__ = ['GuidePreference', 'GuidePreferenceMixin']

class GuidePreference(Base):
    __tablename__ = 'guide_preferences'
    model_id = Column(Integer, nullable = False, primary_key = True)
    model_name = Column(Unicode(255))
    _value = Column('value', Unicode(255))

    def __unicode__(self):
        return u'<GuidePreferences: %s>' % self.value

    @property
    def value(self):
        return json.loads(self._value)

    @value.setter
    def value(self, value):
        self._value = json.dumps(value)

class GuidePreferenceMixin(object):
    """This is mixing to import preferences to models"""

    @property
    def preferences(self):
        pref = session.query(GuidePreference).filter(GuidePreference.model_name == self.__class__.__name__, GuidePreference.model_id == self.id).first()
        return pref.value if pref else {}
