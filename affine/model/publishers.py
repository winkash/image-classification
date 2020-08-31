from affine.model.base import *
from affine.model.preferences import *
from affine.model._sqla_imports import *

__all__ = ['Publisher']

class Publisher(Base, GuidePreferenceMixin):
    __tablename__ = 'publishers'
    id = Column(Integer, nullable = False, primary_key = True)
    name = Column(Unicode(255))

    def line_item_ids(self, campaign_id):
        from affine.model import LineItem
        query = session.query(LineItem.id).filter_by(campaign_id=campaign_id, publisher_id=self.id, archived=False)
        return [li.id for li in query]
