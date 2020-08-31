from datetime import datetime
from sqlalchemy import Column, Integer, DateTime, String, ForeignKey
from sqlalchemy.orm import relation, backref
from sqlalchemy.dialects.mysql import ENUM

from affine.model._sqla_imports import Unicode
from affine.model.globals import countries
from affine.model.base import Base
from affine.model.secondary_tables import channel_positive_keyword_bundles_table, channel_negative_keyword_bundles_table, channel_positive_labels_table, channel_negative_labels_table, channel_positive_user_keywords_table, channel_negative_user_keywords_table

__all__ = ['Channel']


class Channel(Base):
    __tablename__ = 'channels'
    id = Column(Integer, nullable=False, primary_key=True)
    name = Column(Unicode(255), nullable=False)
    advertiser_id = Column(Integer)
    media_partner_id = Column(Integer)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    visibility = Column(Integer, default=0)
    line_item = relation("LineItem", backref='channels')
    country = Column(ENUM(*countries), nullable=False, default='US')
    placement_id = Column(Integer, ForeignKey('placements.id'), default=1)
    created_by = Column(String, default='')
    updated_by = Column(String, default='')

    placement = relation('Placement')
    positive_labels = relation('AbstractLabel', secondary=channel_positive_labels_table)
    negative_labels = relation('AbstractLabel', secondary=channel_negative_labels_table)
    positive_user_keywords = relation(
        'UserKeyword', secondary=channel_positive_user_keywords_table, order_by='UserKeyword.text')
    negative_user_keywords = relation(
        'UserKeyword', secondary=channel_negative_user_keywords_table, order_by='UserKeyword.text')
    positive_keyword_bundles = relation('KeywordBundle', secondary=channel_positive_keyword_bundles_table, order_by='KeywordBundle.name',
                                        backref=backref('positive_channels'))
    negative_keyword_bundles = relation('KeywordBundle', secondary=channel_negative_keyword_bundles_table, order_by='KeywordBundle.name',
                                        backref=backref('negative_channels'))

    def positive_keywords(self):
        """The positive keywords in all bundles for this channel."""
        return [pk.id for pk in self.positive_user_keywords]

    def negative_keywords(self):
        """The negative keywords in all bundles for this channel."""
        return [nk.id for nk in self.negative_user_keywords]
