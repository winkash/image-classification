from logging import getLogger

from affine.model._sqla_imports import *
from affine.model.base import Base, session
from affine.model.web_pages import *


__all__ = ['GoldenHit', 'GoldenHitCandidate', 'VideoHit', 'PageHit', 'BoxHit', 'MTurkBox',
           'ImageHit', 'MTurkImage', 'get_hit_from_hit_id']

logger = getLogger(__name__)

class GoldenHit(Base):
    """A Hit we repost to evaluate workers"""
    __tablename__ = "mturk_golden_hits"
    golden_hit_id = Column(VARCHAR(128), nullable=False, primary_key=True)
    hit_id = Column(VARCHAR(128), nullable=False)

    def __unicode__(self):
        return u'<hit_id:%s, golden_hit_id:%s>' \
                % (self.hit_id, self.golden_hit_id)


class GoldenHitCandidate(Base):
    """A Hit we say can be golden"""
    __tablename__ = "mturk_golden_hit_candidates"
    hit_id = Column(VARCHAR(128), nullable=False, primary_key=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    def __unicode__(self):
        return u'<hit_id:%s, created_at:%s>' \
                % (self.hit_id, self.created_at)


class VideoHit(Base):
    """A task in Mechanical Turk"""
    __tablename__ = "mturk_video_hits"
    label_id = Column(Integer, ForeignKey('labels.id'), nullable = False, primary_key=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable = False, primary_key=True)
    job_id = Column(Integer, nullable=True)
    page_id = Column(Integer, ForeignKey('web_pages.id'), nullable = True)
    hit_id = Column(VARCHAR(128), nullable = False, unique=True)
    outstanding = Column(Boolean, nullable = False, default=True)
    result = Column(Boolean, nullable = True)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    label = relation('Label')
    video = relation('Video')
    page = relation('WebPage')

    def __unicode__(self):
        return u'HIT %s: video_id=%s label=%s' % (self.hit_id, self.video_id, self.label.name)

    @classmethod
    def update_mturk_results(cls, mt_results):
        for hit_id, video_id, label_id, result in mt_results:
            """ take ingested results from MTurk and update results on the DB """
            vh = VideoHit.query.filter_by(hit_id = hit_id).first()
            if vh is None:
                logger.warn("Hit not found %s", hit_id)
            else:
                vh.result = result
                vh.outstanding = False
        session.flush()

    @classmethod
    def get_potential_golden_hit_candidates(cls):
        """Get hits which could be good golden hit candidates.

        Returns:
            VideoHit query.
        """
        query = cls.query.filter(cls.outstanding == False, cls.result != None)
        return query.order_by(cls.timestamp.desc())


class PageHit(Base):
    """A task in Mechanical Turk"""
    __tablename__ = "mturk_page_hits"
    label_id = Column(Integer, ForeignKey('labels.id'), nullable=False, primary_key=True)
    page_id = Column(Integer, ForeignKey('web_pages.id'), nullable=False, primary_key=True)
    job_id = Column(Integer, nullable=True)
    hit_id = Column(VARCHAR(128), nullable=False, unique=True)
    outstanding = Column(Boolean, nullable=False, default=True)
    result = Column(Boolean, nullable=True, default=None)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    label = relation('Label')
    page = relation('WebPage')

    def __unicode__(self):
        return u'HIT %s: page_id=%s label=%s' % (self.hit_id, self.page_id, self.label.name)

    @classmethod
    def update_mturk_results(cls, mt_results):
        """ take ingested results from MTurk and update results on the DB """
        for hit_id, page_id, label_id, result in mt_results:
            ph = PageHit.query.filter_by(hit_id=hit_id).first()
            if ph is None:
                logger.warn("Hit not found %s", hit_id)
            else:
                ph.result = result
                ph.outstanding = False
        session.flush()

    @classmethod
    def get_potential_golden_hit_candidates(cls):
        """Get hits which could be good golden hit candidates.

        Returns:
            PageHit query.
        """
        query = cls.query.filter(cls.outstanding == False, cls.result != None)
        return query.order_by(cls.timestamp.desc())


class BoxHit(Base):
    """A task in Mechanical Turk"""
    __tablename__ = "mturk_box_hits"
    id = Column(Integer, nullable=False, primary_key=True)
    hit_id = Column(VARCHAR(128), nullable = False)
    outstanding = Column(Boolean, nullable = False, default=True)
    training_job_id = Column(Integer, ForeignKey('mturk_training_jobs.id'))
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    job = relation('TrainingJob', backref='hits')

    def __unicode__(self):
        box_string = [box.__unicode__() for box in self.boxes]
        box_string = ",".join(box_string)
        return u'Box HIT %s: boxes=%s' % (self.hit_id, box_string)

    @classmethod
    def get_potential_golden_hit_candidates(cls, min_num_boxes=20):
        """Get hits which could be good golden hit candidates.

        Args:
            min_num_boxes: Optional, the minimum number of boxes per box hit.

        Returns:
            BoxHit query.
        """
        query = cls.query.filter(cls.outstanding == False,
                                 ~cls.boxes.any(MTurkBox.result == None))
        query = query.join(MTurkBox, MTurkBox.box_hit_id == cls.id)
        query = query.group_by(cls.id).having(func.count(MTurkBox.id) >= min_num_boxes)
        return query.order_by(cls.timestamp.desc())


class MTurkBox(Base):
    """A single box on a HIT. Each BoxHit has multiple MturkBoxes"""
    __tablename__ = "mturk_training_boxes"
    id = Column(Integer, nullable=False, primary_key=True, autoincrement = True)
    box_id = Column(Integer, ForeignKey('boxes.id'), nullable = False)
    box_hit_id = Column(Integer, ForeignKey('mturk_box_hits.id'), nullable=False)
    label_id = Column(Integer, ForeignKey('labels.id'), nullable=False)
    result = Column(Boolean)

    hit = relation('BoxHit', backref='boxes')
    box = relation('Box')
    label = relation('Label')

    def __unicode__(self):
        return u'MTurkBox for box:%s Label:%s on HIT %s' % (str(self.box_id), self.label_id, self.box_hit_id)

    @classmethod
    def update_mturk_results(cls, mt_results):
        for hit_id, box_id, label_id, result in mt_results:
            mb = cls.query.filter_by(box_id = box_id, label_id = label_id).first()
            if not mb:
                logger.warn('MTurkBox not found for box_id:%s, label_id:%s and BoxHit.hit_id:%s' %(box_id, label_id, hit_id))
            else:
                mb.result = result
                BoxHit.query.filter_by(hit_id=hit_id).update({"outstanding":
                    False}, synchronize_session=False)
        session.flush()


class ImageHit(Base):
    """ A Hit consisting of multiple Clickable images """
    __tablename__ = "mturk_image_hits"
    id = Column(Integer, nullable=False, primary_key=True)
    hit_id = Column(VARCHAR(128), nullable = False)
    outstanding = Column(Boolean, nullable = False, default=True)
    mturk_image_job_id = Column(Integer, ForeignKey('mturk_image_jobs.id'))
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    def __unicode__(self):
        return u'Image HIT: %s, outstanding: %s' % (self.hit_id, self.outstanding)

    @classmethod
    def get_potential_golden_hit_candidates(cls, min_num_images=20):
        """Get hits which could be good golden hit candidates.

        Args:
            min_num_images: Optional, the minimum number of images per image hit.

        Returns:
            ImageHit query.
        """
        query = cls.query.filter(cls.outstanding == False,
                                 ~cls.images.any(MTurkImage.result == None))
        query = query.join(MTurkImage, MTurkImage.image_hit_id == cls.id)
        query = query.group_by(cls.id).having(func.count(MTurkImage.id) >= min_num_images)
        return query.order_by(cls.timestamp.desc())


class MTurkImage(Base):
    """ A single image put on MTurk. Part of an ImageHIT """
    __tablename__ = "mturk_images"
    id = Column(Integer, nullable=False, primary_key=True, autoincrement = True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False)
    timestamp = Column(Integer, nullable=False)
    image_hit_id = Column(Integer, ForeignKey('mturk_image_hits.id'), nullable=False)
    label_id = Column(Integer, ForeignKey('labels.id'), nullable=False)
    result = Column(Boolean)

    hit = relation('ImageHit', backref='images')
    video = relation('Video')
    label = relation('Label')

    def __unicode__(self):
        return u'MTurkImage for video_id:%s, timestamp:%s, label_id:%s on ImageHit:%s' %(self.video_id, self.timestamp, self.label_id, self.image_hit_id)

    @classmethod
    def update_mturk_results(cls, mt_results):
        for hit_id, video_id, timestamp, label_id, result in mt_results:
            mi = cls.query.filter_by(video_id=video_id, timestamp=timestamp, label_id=label_id).first()
            if mi is None:
                logger.warn("MTurkImage not found for video_id:%s, timestamp%s, ImageHit.hit_id:%s" %(video_id, timestamp, hit_id))
            else:
                mi.result = result
                mi.hit.outstanding = False
        session.flush()


HIT_TYPES = [VideoHit, PageHit, ImageHit, BoxHit]

def get_hit_from_hit_id(hit_id):
    """Get hit from its hit_id"""
    for cls in HIT_TYPES:
        hit = cls.query.filter_by(hit_id=hit_id).first()
        if hit:
            return hit
