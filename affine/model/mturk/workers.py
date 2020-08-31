from datetime import datetime

from affine.model._sqla_imports import *
from affine.model.base import Base, session
from affine.aws.mturk import MTurkUtils

BLOCK_REASON = "We regret to inform you that you've been blocked. \
Our automated system has found that you've committed more \
errors than acceptable when answering our HITs. We apologize for any \
inconvenience this may cause. Thank you."

__all__ = ['MTurkWorker']


class MTurkWorker(Base):
    __tablename__ = 'mturk_workers'
    worker_id = Column(CHAR, nullable=False, primary_key=True)
    yes_count = Column(Integer, nullable=False, default=0)
    no_count = Column(Integer, nullable=False, default=0)
    num_minority = Column(Integer, nullable=False, default=0)
    time_elapsed = Column(Integer, nullable=False, default=0)
    num_golden_error = Column(Integer, nullable=False, default=0)
    num_golden = Column(Integer, nullable=False, default=0)
    blocked_since = Column(DateTime)

    @classmethod
    def get_or_create(cls, worker_id):
        wk = MTurkWorker.query.filter_by(worker_id=worker_id).scalar()
        if wk is None:
            wk = MTurkWorker(worker_id=worker_id)
            session.flush()
        return wk

    def block(self, reason=BLOCK_REASON):
        MTurkUtils.block_worker(self.worker_id, reason)
        self.blocked_since = datetime.utcnow()
        session.flush()

    def unblock(self, reason=""):
        MTurkUtils.unblock_worker(self.worker_id, reason)
        self.blocked_since = None
        session.flush()

    def __unicode__(self):
        return u'<worker_id:%s, yes_count:%s, no_count:%s, num_minority:%s, '\
                'time_elapsed:%s, num_golden_error:%s, num_golden:%s, blocked_since:%s>'\
                % (self.worker_id, self.yes_count, self.no_count,
                self.num_minority, self.time_elapsed, self.num_golden_error,
                self.num_golden, self.blocked_since)
