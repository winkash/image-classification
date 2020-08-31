from datetime import datetime

from affine.model.base import *
from affine.model.preferences import *
from affine.model._sqla_imports import *
from affine.model.campaigns import CampaignPropertiesMixin

__all__ = ['LineItemGroup']


class LineItemGroup(Base, CampaignPropertiesMixin, GuidePreferenceMixin):
    __tablename__ = 'line_item_groups'
    id = Column(Integer, nullable = False, primary_key = True)
    name = Column(Unicode(255))
    campaign_id = Column(Integer, ForeignKey('campaigns.id'), nullable = False)
    cost_model = Column(VARCHAR(6), server_default='CPM')
    impression_target = Column(Integer, nullable=True)
    completion_view_target = Column(Integer)
    display_timezone = Column(VARCHAR(128), server_default='UTC')
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    campaign = relation("Campaign", backref=backref('line_item_groups', cascade='all,delete-orphan'))

    @property
    def line_item_ids(self):
        from affine.model import LineItem
        query = session.query(LineItem.id).filter_by(line_item_group_id=self.id, archived=False)
        return [li.id for li in query]


    @property
    def start_date(self):
        from affine.model import LineItem
        query = session.query(func.min(LineItem.start_date)).filter(LineItem.line_item_group_id==self.id).first()
        for start_date in query:
            return start_date
        return None

    @property
    def end_date(self):
        from affine.model import LineItem
        query = session.query(func.max(LineItem.end_date)).filter(LineItem.line_item_group_id==self.id).first()
        for end_date in query:
            return end_date
        return None
