from affine.model._sqla_imports import *
from affine.model.base import *
from affine.model import Label

__all__ = ['NamedEntity']

class NamedEntity(Base):
    __tablename__ = "named_entities"

    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128, charset='utf8', convert_unicode=True, collation="utf8_bin"), nullable=False)
    entity_type = Column(Enum('person', 'organization'), nullable=False)
    label_id = Column(Integer, ForeignKey('labels.id'), nullable=False)
    fb_id = Column(VARCHAR(128))

    label = relation('Label')

    def __unicode__(self):
        return u'<Named Entity:%s, Label:%s, Type:%s>' %(self.name, self.label.name, self.entity_type)

    @classmethod
    def get_or_create(cls, name, entity_type, label_id, fb_id=None):
        named_entity = cls.query.filter_by(name=name, entity_type=entity_type, label_id=label_id).first()
        if not named_entity:
            named_entity = cls(name=name, entity_type=entity_type, label_id=label_id, fb_id=fb_id)
            session.flush()
        return named_entity
