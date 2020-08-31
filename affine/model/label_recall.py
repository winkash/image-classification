from datetime import datetime, timedelta
from affine.model._sqla_imports import *
from affine.model.base import *

__all__ = ['LabelRecall']

class LabelRecall(Base):
    __tablename__ = 'label_recall'

    id = Column(Integer, primary_key=True)
    created = Column(DateTime, nullable=False, default=datetime.utcnow)
    label_id = Column(Integer, ForeignKey('labels.id'), nullable=False)
    recall = Column(Float, nullable=False)

    @classmethod
    def get_latest_recall(cls, label_id):
        """ Returns the latest recall value for a particular label-id """
        if label_id is None:
            raise TypeError('Invalid label-id')
        recall_result = cls.query.filter_by(label_id=label_id).order_by(LabelRecall.created.desc()).first()
        if not recall_result:
            return None
        return recall_result.recall

    @classmethod
    def get_recall(cls, label_id, ip_date):
        """ Returns the latest recall value on a particular date for a label-id """
        if label_id is None:
            raise TypeError('Invalid label-id')

        end_date = ip_date + timedelta(days=1)
        query = cls.query.filter_by(label_id=label_id).filter(cls.created>=ip_date, cls.created<end_date)
        recall_result = query.order_by(cls.created.desc()).first()
        if not recall_result:
            return None
        return recall_result.recall

def __unicode__(self):
        return "<LabelRecall(%s) label_id(%s) recall(%f)>" %(self.id, self.label_id, self.recall)
