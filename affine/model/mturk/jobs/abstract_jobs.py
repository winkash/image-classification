from datetime import datetime
from uuid import uuid1
from affine.model.base import *
from affine.model._sqla_imports import *

__all__ = ['AbstractMTurkJob', 'AbstractMTurkLabelJob']

class AbstractMTurkJob(Base):
    """ Abstract Base Class for submitting MTurk HITs
    """
    __tablename__ = 'abstract_mturk_jobs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Unicode(128), unique=True, nullable=False)
    start_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=True)
    max_hits_per_submission = Column(Integer, nullable=False)
    max_hits = Column(Integer, nullable=False)
    _cls = Column('cls', String(50), nullable=False)
    __mapper_args__ = dict(polymorphic_on=_cls, with_polymorphic='*')
    __table_args__ = (
        UniqueConstraint('cls', 'name', name='uniq_name_per_job_type'),
    )


class AbstractMTurkLabelJob(AbstractMTurkJob):
    """ Abstract Job class for submitting HITs for Labels
    """
    label_id = Column(Integer, ForeignKey('abstract_labels.id'))

    @classmethod
    def create(cls, label_id, max_hits_per_submission, max_hits, name=None, **kwargs):
        if name is None:
            name = '%s_%s_%s' %(label_id, cls.result_table.__table__.name, uuid1())
        job = cls(label_id=label_id, name=name, max_hits=max_hits,
                  max_hits_per_submission=max_hits_per_submission, **kwargs)
        session.flush()
        return job

    def _get_limit(self, hits_per_submission, hits_submitted, max_hits):
        return int(min(max(max_hits - hits_submitted, 0), hits_per_submission))
