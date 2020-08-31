from datetime import datetime
from collections import defaultdict

from affine.model.base import *
from affine.model.preferences import *
from affine.model.users import User, MediaPartner
from affine.model.campaigns import CampaignMixin
from affine.model.secondary_tables import line_item_media_partners_table, line_item_negative_labels_table
from affine.model._sqla_imports import *

__all__ = ['LineItem']


class LineItem(Base, CampaignMixin, GuidePreferenceMixin):
    __tablename__ = "line_items"
    id = Column(Integer, primary_key=True)
    campaign_id = Column(Integer, ForeignKey('campaigns.id'), nullable=False)
    channel_id = Column(Integer, ForeignKey('channels.id'))
    media_partner_id = Column(Integer)
    archived = Column(Boolean, nullable=False, default=False)
    impression_target = Column(Integer)
    cost_model = Column(VARCHAR(6), server_default='CPM')
    display_timezone = Column(VARCHAR(128), server_default='UTC')
    completion_view_target = Column(Integer)
    line_item_group_id = Column(Integer, ForeignKey('line_item_groups.id'), nullable=False)
    publisher_id = Column(Integer, ForeignKey('publishers.id'), nullable=False)
    is_diagnostic = Column(Boolean, nullable=False, default=False)
    is_contextual = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by = Column(String, default='')
    updated_at = Column(Timestamp, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String, default='')
    campaign = relation('Campaign', backref=backref('line_items', cascade='all,delete-orphan'))
    channel = relation('Channel', enable_typechecks=False, backref='line_items')
    media_partners = relation('MediaPartner', secondary=line_item_media_partners_table, backref='line_items',
                             order_by='MediaPartner.name')
    line_item_group = relation('LineItemGroup', backref=backref('line_items'))
    publisher = relation('Publisher', backref=backref('line_items'))
    negative_labels = relation('Label', secondary=line_item_negative_labels_table)

    def get_flight_data(self):
        start_date = self.start_date
        end_date = self.end_date
        camp_start = self.campaign.start_date
        camp_end = self.campaign.end_date
        if start_date is None or (camp_start is not None and camp_start > start_date):
            start_date = camp_start
        if end_date is None or (camp_end is not None and camp_end < end_date):
            end_date = camp_end
        return start_date, end_date

    def find_channels_labels(self):
        if self.channel:
            return self.channel.positive_labels + self.channel.negative_labels
        else:
            return []

    def authorized_keys(self):
        """Everyone who is authorized for the given line item

        Returns a dict where the keys are auth keys and the values are media partner names.
        """
        query = session.query(User.auth_key, MediaPartner.name)
        query = query.filter(User.auth_key != None).filter(User.auth_key != '')
        query = query.join(User.media_partners)
        query = query.join(MediaPartner.line_items).filter_by(id=self.id)

        ret = defaultdict(list)
        for key, mp_name in query:
            ret[key].append(mp_name)
        return dict(ret)

    @property
    def line_item_ids(self):
        return [self.id]
