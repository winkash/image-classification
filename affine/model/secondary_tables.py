from affine.model.base import metadata
from affine.model._sqla_imports import *

bundle_keywords_table = Table('bundle_keywords', metadata,
    Column('bundle_id', Integer, ForeignKey('keyword_bundles.id'), nullable=False),
    Column('user_keyword_id', Integer, ForeignKey('user_keywords.id'), nullable=False),
    UniqueConstraint('bundle_id', 'user_keyword_id', name='bundle_user_keyword_idx'))

channel_positive_labels_table = Table('channel_positive_labels', metadata,
    Column('id', Integer, nullable=True, primary_key = True),
    Column('channel_id', Integer, ForeignKey('channels.id'), nullable = False),
    Column('label_id', Integer, ForeignKey('abstract_labels.id'), nullable = False))

channel_negative_labels_table = Table('channel_negative_labels', metadata,
    Column('id', Integer, nullable=True, primary_key = True),
    Column('channel_id', Integer, ForeignKey('channels.id'), nullable = False),
    Column('label_id', Integer, ForeignKey('abstract_labels.id'), nullable = False))

channel_positive_user_keywords_table = Table('channel_positive_user_keywords', metadata,
    Column('channel_id', Integer, ForeignKey('channels.id'), nullable = False, primary_key=True),
    Column('user_keyword_id', Integer, ForeignKey('user_keywords.id'), nullable = False, primary_key=True),
    UniqueConstraint('channel_id', 'user_keyword_id', name='channel_user_keyword_idx'))

channel_negative_user_keywords_table = Table('channel_negative_user_keywords', metadata,
    Column('channel_id', Integer, ForeignKey('channels.id'), nullable = False, primary_key=True),
    Column('user_keyword_id', Integer, ForeignKey('user_keywords.id'), nullable = False, primary_key=True),
    UniqueConstraint('channel_id', 'user_keyword_id', name='channel_user_keyword_idx'))

channel_positive_keyword_bundles_table = Table('channel_positive_keyword_bundles', metadata,
    Column('channel_id', Integer, ForeignKey('channels.id'), nullable = False, primary_key=True),
    Column('keyword_bundle_id', Integer, ForeignKey('keyword_bundles.id'), nullable = False, primary_key=True),
    UniqueConstraint('channel_id', 'keyword_bundle_id', name='channel_keyword_bundle_idx'))

channel_negative_keyword_bundles_table = Table('channel_negative_keyword_bundles', metadata,
    Column('channel_id', Integer, ForeignKey('channels.id'), nullable = False, primary_key=True),
    Column('keyword_bundle_id', Integer, ForeignKey('keyword_bundles.id'), nullable = False, primary_key=True),
    UniqueConstraint('channel_id', 'keyword_bundle_id', name='channel_keyword_bundle_idx'))

line_item_media_partners_table = Table('line_item_media_partners', metadata,
    Column('line_item_id', Integer, ForeignKey('line_items.id'), nullable=False),
    Column('media_partner_id', Integer, ForeignKey('media_partners.id'), nullable=False))

line_item_negative_labels_table = Table('line_item_negative_labels', metadata,
    Column('line_item_id', Integer, ForeignKey('line_items.id'), nullable=False, primary_key=True),
    Column('label_id', Integer, ForeignKey('abstract_labels.id'), nullable=False, primary_key=True),
    UniqueConstraint('line_item_id', 'label_id', name='line_item_negative_labels_idx'))

__all__ = [name for name, value in locals().items() if isinstance(value, Table)]
