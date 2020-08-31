from affine.model.base import *
from affine.model._sqla_imports import *

__all__ = ['Settings']

class Settings(Base):
    """
        Represent a name-value pair for users to configure a setting to True/False
        Currently its intended to be used for specifying if the running vcr instance
        to save videos & images or not
    """
    __tablename__ = 'settings'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    value = Column(Boolean, nullable=False)

    @classmethod
    def set_value(cls, name, value):
        """ set a boolean True/False value for a given settings name """
        instance = cls.by_name(name)
        if not instance:
            instance = cls(name = name, value = value)
        else:
            instance.value = value
        session.flush()
        return instance.value

    @classmethod
    def get_value(cls, name):
        """ retrieve the boolean result for a given settings name, None if it was never set """
        instance = cls.by_name(name)
        if instance:
            return instance.value

    def __unicode__(self):
        return u"Setting(name = '%s', value = %s)" % (self.name, self.value)
