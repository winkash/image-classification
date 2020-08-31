from datetime import datetime
from hashlib import sha1
from uuid import uuid1

from sqlalchemy.ext.hybrid import hybrid_property

from affine.model.base import *
from affine.model import GuidePreferenceMixin
from affine.model._sqla_imports import *

__all__ = ['Advertiser', 'User', 'MediaPartner',
           'Affiliation', 'MediaPartnerAffiliation', 'AdvertiserAffiliation']


class User(Base, GuidePreferenceMixin):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(Unicode(255), nullable=False, default='')
    auth_key = Column(Unicode(255))
    encrypted_password = Column(VARCHAR(128), nullable=False, default='')
    reset_password_token = Column(VARCHAR(255))
    reset_password_sent_at = Column(DateTime)
    remember_created_at = Column(DateTime)
    allowed_access = Column(Boolean, default=True, nullable=False)
    sign_in_count = Column(Integer, default=0)
    current_sign_in_at = Column(DateTime)
    last_sign_in_at = Column(DateTime)
    current_sign_in_ip = Column(VARCHAR(255))
    last_sign_in_ip = Column(VARCHAR(255))
    time_zone = Column(VARCHAR(255), default='Pacific Time (US & Canada)')
    admin = Column(Boolean, nullable=False, default=False)
    permission = Column(VARCHAR(255), default='advertiser')
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, default='')
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String, default='')

    media_partners = relation(
        'MediaPartner', secondary=lambda : Affiliation.__table__,
        primaryjoin='User.id == Affiliation.user_id',
        secondaryjoin='''and_(Affiliation._cls == 'MediaPartner',
                              MediaPartner.id == Affiliation._affiliable_id)''',
        foreign_keys=lambda : [Affiliation.user_id, Affiliation._affiliable_id],
        order_by=lambda: MediaPartner.name,
        viewonly=True,
        backref=backref('users', viewonly=True))

    advertisers = relation(
        'Advertiser', secondary=lambda : Affiliation.__table__,
        primaryjoin='User.id == Affiliation.user_id',
        secondaryjoin='''and_(Affiliation._cls == 'Advertiser',
                              Advertiser.id == Affiliation._affiliable_id)''',
        foreign_keys=lambda : [Affiliation.user_id, Affiliation._affiliable_id],
        order_by=lambda: Advertiser.name,
        viewonly=True,
        backref=backref('users', viewonly=True))

    def __unicode__(self):
        return u'<User: %s>' % self.email

    @property
    def is_admin(self):
        return self.permission == 'admin' or self.permission == 'general' or self.admin

    @property
    def line_item_ids(self):
        if self.is_admin:
            from affine.model import LineItem
            li_ids = session.query(LineItem.id).filter(LineItem.archived == False).all()
            return [id for subl in li_ids for id in subl]
        else:
            li_ids = [adv.line_item_ids for adv in self.advertisers]
            return [id for subl in li_ids for id in subl]

    @classmethod
    def by_email(cls, email):
        return cls.query.filter_by(email=email).first()

    def generate_auth_key(self):
        self.auth_key = sha1(str(uuid1())).hexdigest()
        session.flush()


class MediaPartner(Base):
    __tablename__ = 'media_partners'
    id = Column(Integer, primary_key=True)
    name = Column(Unicode(64), nullable=False, unique=True)
    password = Column(VARCHAR(40))
    auth_key = Column(VARCHAR(100))
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by = Column(String, default='')
    updated_at = Column(Timestamp, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String, default='')


class Advertiser(Base, GuidePreferenceMixin):
    __tablename__ = 'advertisers'
    id = Column(Integer, primary_key=True)
    name = Column(Unicode(255), nullable=False, unique=True)
    media_partner_id = Column(Integer)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String, default='')
    updated_by = Column(String, default='')

    @property
    def line_item_ids(self):
        from affine.model import LineItem
        query = session.query(LineItem.id).filter_by(archived=False).join(LineItem.campaign).filter_by(archived=False, advertiser_id=self.id)
        return [li.id for li in query]


class Affiliation(Base):
    """Represent associations between users (individuals) and media partners or advertisers"""
    __tablename__ = 'affiliations'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    _affiliable_id = Column('affiliable_id', Integer, nullable=False)
    _cls = Column('affiliable_type', Enum('Advertiser','MediaPartner'), nullable=False)

    __mapper_args__ = dict(polymorphic_on=_cls)

    user = relationship("User", backref='affiliations')


class AdvertiserAffiliation(Affiliation):
    __mapper_args__ = dict(polymorphic_identity='Advertiser')

    @hybrid_property
    def advertiser_id(self):
        return self._affiliable_id

    @advertiser_id.setter
    def advertiser_id(self, value):
        self._affiliable_id = value

    advertiser = relation(
        'Advertiser',
        primaryjoin=Affiliation._affiliable_id == Advertiser.id,
        foreign_keys=[Affiliation._affiliable_id],
        backref=backref('affiliations', cascade='all'))

    _user = relationship("User", backref='advertiser_affiliations')


class MediaPartnerAffiliation(Affiliation):
    __mapper_args__ = dict(polymorphic_identity='MediaPartner')

    @hybrid_property
    def media_partner_id(self):
        return self._affiliable_id

    @media_partner_id.setter
    def media_partner_id(self, value):
        self._affiliable_id = value

    media_partner = relation(
        'MediaPartner',
        enable_typechecks=False,
        primaryjoin=Affiliation._affiliable_id == MediaPartner.id,
        foreign_keys=[Affiliation._affiliable_id],
        backref=backref('affiliations', cascade='all'))

    _user = relationship("User", enable_typechecks=False,backref='media_partner_affiliations')
