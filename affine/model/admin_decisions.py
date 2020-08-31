from datetime import datetime
from affine.model.base import *
from affine.model._sqla_imports import *

__all__ = ['AdminAppLabelResult', 'AdminVideoLabelResult', 'AdminWebPageLabelResult', 'AdminVideoAd']

class AdminWebPageLabelResult(Base):
    """A boolean result of a specific page for a specific label"""
    __tablename__ = "admin_web_page_label_results"
    user_id = Column(Integer, ForeignKey('users.id'), nullable = False)
    page_id = Column(Integer, ForeignKey('web_pages.id'), nullable = False, primary_key=True)
    label_id = Column(Integer, ForeignKey('labels.id'), nullable = False, primary_key=True)
    result = Column(Boolean, nullable=False)
    last_modified = Column(DateTime, default=datetime.utcnow)
    user = relation('User')
    page = relation('WebPage', backref=backref('admin_label_results', cascade='all,delete-orphan'))
    label = relation('Label', backref=backref('admin_web_page_results', passive_deletes=True))

    def __unicode__(self):
        label_name = self.label_id
        if self.label:
            label_name = self.label.name
        return u'<admin web page result (%s, %s, %s)>' % (self.page_id, label_name, self.result)

    @classmethod
    def get_result(cls, page_id, label_id):
        avlr = cls.query.filter_by(page_id = page_id,
                                   label_id = label_id).first()
        if avlr:
            return avlr.result

    @classmethod
    def set_result(cls, page_id, label_id, user_id, result):
        if result is None:
            cls.clear_result(page_id, label_id)
            return

        awplr = cls.query.filter_by(page_id=page_id, label_id=label_id).first()
        if awplr is not None:
            if awplr.result == result:
                return
            else:
                awplr.result = result
                awplr.last_modified = datetime.utcnow()
                awplr.user_id = user_id
        else:
            awplr = cls(page_id=page_id, label_id=label_id, user_id=user_id, result=result)
        session.flush()

    @classmethod
    def clear_result(cls, page_id, label_id):
        cls.query.filter_by(page_id = page_id, label_id = label_id).delete()


class AdminVideoLabelResult(Base):
    """A boolean result of a specific video for a specific label"""
    __tablename__ = "admin_video_label_results"
    user_id = Column(Integer, ForeignKey('users.id'), nullable = False)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable = False, primary_key=True)
    label_id = Column(Integer, ForeignKey('labels.id'), nullable = False, primary_key=True)
    result = Column(Boolean, nullable=False)
    processed = Column(Boolean, nullable=False, default=False)
    last_modified = Column(DateTime, default=datetime.utcnow)
    user = relation('User')
    video = relation('Video', backref=backref('admin_label_results', cascade='all,delete-orphan'))
    label = relation('Label', backref=backref('admin_video_results', passive_deletes=True))

    def __unicode__(self):
        label_name = self.label_id
        if self.label:
            label_name = self.label.name
        return u'<admin video result (%s, %s, %s)>' % (self.video_id, label_name, self.result)

    @classmethod
    def get_result(cls, video_id, label_id):
        avlr = cls.query.filter_by(video_id = video_id,
                                   label_id = label_id).first()
        if avlr:
            return avlr.result

    @classmethod
    def set_result(cls, video_id, label_id, user_id, result):
        if result is None:
            cls.clear_result(video_id, label_id)
            return

        avlr = cls.query.filter_by(video_id=video_id, label_id=label_id).first()
        if avlr is not None:
            if avlr.result == result:
                return
            else:
                avlr.result = result
                avlr.last_modified = datetime.utcnow()
                avlr.processed = False
                avlr.user_id = user_id
        else:
            avlr = cls(video_id=video_id, label_id=label_id, user_id=user_id, result=result)
        session.flush()

    @classmethod
    def clear_result(cls, video_id, label_id):
        cls.query.filter_by(video_id = video_id, label_id = label_id).delete()
    

class AdminVideoAd(Base):
    __tablename__ = "admin_video_ads"
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    created = Column(Timestamp, nullable=False, server_default=func.now())
    video = relation('Video')
    user = relation('User')

    def __unicode__(self):
        return u'<AdminVideoAd(video_id=%s)>' % ( self.video_id)

    @classmethod
    def get_result_by_checksum(cls, video_checksum):
        return session.query(cls.video_id).join(cls.video).filter_by(checksum=video_checksum).scalar() is not None

    @classmethod
    def add(cls, video_id):
        # Persist any unflushed changed so they don't get lost if we hit an
        # IntegrityError and rollback
        session.flush()
        try:
            cls(video_id=video_id)
            session.flush()
        except IntegrityError as e:
            if 'Duplicate entry' not in str(e):
                raise

    @classmethod
    def set_result(cls, video_id, user_id):
        admin_ad = cls.query.filter_by(video_id=video_id).first()
        if admin_ad is not None:
            return
        cls.create(video_id=video_id, user_id=user_id)


class AdminAppLabelResult(Base):
    """A boolean result of a specific app for a specific label"""
    __tablename__ = "admin_app_label_results"
    user_id = Column(Integer, ForeignKey('users.id'), nullable = False)
    app_id = Column(Integer, ForeignKey('apps.id'), nullable = False, primary_key=True)
    label_id = Column(Integer, ForeignKey('abstract_labels.id'), nullable = False, primary_key=True)
    result = Column(Boolean, nullable=False)
    last_modified = Column(DateTime, default=datetime.utcnow)
    user = relation('User')
    app = relation('App', backref=backref('admin_label_results', cascade='all,delete-orphan'))
    label = relation('AppLabel', backref=backref('admin_app_results', passive_deletes=True))

    def __unicode__(self):
        label_name = self.label_id
        if self.label:
            label_name = self.label.name
        return u'<admin app result (%s, %s, %s)>' % (self.app_id, label_name, self.result)

    @classmethod
    def get_result(cls, app_id, label_id):
        aalr = cls.query.filter_by(app_id = app_id,
                                   label_id = label_id).first()
        if aalr:
            return aalr.result

    @classmethod
    def set_result(cls, app_id, label_id, user_id, result):
        if result is None:
            cls.clear_result(app_id, label_id)
            return

        aalr = cls.query.filter_by(app_id=app_id, label_id=label_id).first()
        if aalr is not None:
            if aalr.result == result:
                return
            else:
                aalr.result = result
                aalr.last_modified = datetime.utcnow()
                aalr.processed = False
                aalr.user_id = user_id
        else:
            aalr = cls(app_id=app_id, label_id=label_id, user_id=user_id, result=result)
        session.flush()

    @classmethod
    def clear_result(cls, app_id, label_id):
        cls.query.filter_by(app_id = app_id, label_id = label_id).delete()


