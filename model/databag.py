from util import PickleMixin
import numpy as np

__all__ = ['DataBag']


class DataBag(PickleMixin):
    """Class to encapsulate training/testing boxes, signatures, labels"""
    def __init__(self, feats, labs, box_ids=None):
        super(DataBag, self).__init__()
        self.feats = feats
        self.labs = labs
        self.box_ids = box_ids or []

    def __add__(self, other):
        db = DataBag([], [], [])
        db.feats = np.concatenate((np.asarray(self.feats),
            np.asarray(other.feats)))
        db.labs = self.labs + other.labs
        db.box_ids = self.box_ids + other.box_ids
        return db
