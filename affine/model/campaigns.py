from datetime import datetime, timedelta
from affine.model.base import *
from affine.model.preferences import *
from affine.model._sqla_imports import *

__all__ = ['Campaign']


class CampaignPropertiesMixin(object):
    @property
    def metric_target(self):
        if self.cost_model == 'CPM':
            return self.impression_target
        elif self.cost_model == 'CPCV':
            return self.completion_view_target

    @property
    def dates(self):
        if self.end_date is None:
            return None
        dates = []
        cur_date = self.start_date.date()
        day = timedelta(days=1)
        while cur_date <= self.end_date.date():
            dates.append(cur_date)
            cur_date += day
        return dates


class CampaignMixin(CampaignPropertiesMixin):
    name = Column(Unicode(255))
    cost_model = Column(VARCHAR(6), server_default='CPM')
    impression_target = Column(Integer)
    completion_view_target = Column(Integer)
    budget_cents = Column(Integer)
    budget_cents_per_day = Column(Integer)

    status = Column(Unicode(255), server_default='active')
    start_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_date = Column(DateTime)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String, default='')
    updated_by = Column(String, default='')
    is_diagnostic = Column(Boolean, nullable=False, default=False)
    archived = Column(Boolean, nullable=False, default=False)
    display_timezone = Column(VARCHAR(128), server_default='UTC')


class Campaign(Base, CampaignMixin, GuidePreferenceMixin):
    __tablename__ = 'campaigns'
    id = Column(Integer, nullable = False, primary_key = True)
    rfp_url = Column(URL)
    advertiser_id = Column(Integer, ForeignKey('advertisers.id'), nullable = False)

    advertiser = relationship("Advertiser", backref=backref('campaigns', cascade='all,delete-orphan'))

    @property
    def line_item_ids(self):
        from affine.model import LineItem
        query = session.query(LineItem.id).filter_by(archived=False, campaign_id=self.id)
        return [li.id for li in query]

    @property
    def publishers(self):
        from affine.model import LineItem, Publisher
        return session.query(Publisher).join(LineItem).filter(LineItem.archived==False, LineItem.campaign_id==self.id, LineItem.publisher_id != None).all()
