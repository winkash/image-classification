from datetime import datetime, timedelta
import hashlib

from affine.model.base import *
from affine.model.detection import DomainNameDetector, AbstractClassifier, \
    AbstractDetector, AbstractTextDetector
from affine.model.labels import Label, Keyword, WeightedLabel, WeightedKeyword,\
    WeightedClfTarget
from affine.model.classifier_target_labels import ClassifierTarget
from affine.model._sqla_imports import *
from affine.model.url_blacklist import RotatingContentPage

__all__ = ['LabelHash']


class LabelHash(Base):
    __tablename__ = "label_hash_tags"
    label_id = Column(Integer, ForeignKey('abstract_labels.id', ondelete="cascade"), nullable=False, primary_key=True)
    hash_tag = Column(CHAR(length=40))
    hash_updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    latest_detector = Column(DateTime)
    latest_text_detector = Column(DateTime)

    label = relation('AbstractLabel', backref=backref('label_hash', uselist=False, cascade='all,delete-orphan'))

    # Timestamps used by query service
    qs_hash_updated_at = Column(DateTime)
    qs_latest_detector = Column(DateTime)
    qs_latest_text_detector = Column(DateTime)

    # How soon timestamps can be promoted to be used by query service
    min_age_for_hash_updated_at = timedelta(days=7)
    min_age_for_latest_detector = timedelta(days=30)
    min_age_for_latest_text_detector = timedelta(days=30)

    def __unicode__(self):
        return "<Label:%s hash tag:%s>" %(self.label_id, self.hash_tag)

    def promote_timestamps(self):
        timestamp_names = ['hash_updated_at', 'latest_detector', 'latest_text_detector']

        # check if we should promote the timestamps
        for timestamp_name in timestamp_names:
            timestamp = getattr(self, timestamp_name)
            min_age = getattr(self, 'min_age_for_' + timestamp_name)
            max_dt = datetime.utcnow() - min_age
            if timestamp is not None and timestamp > max_dt:
                return

        # promote the timestamps
        for timestamp_name in timestamp_names:
            timestamp = getattr(self, timestamp_name)
            setattr(self, 'qs_' + timestamp_name, timestamp)

    def update(self, hash_lookup=None):
        old_hash_tag = self.hash_tag
        self.hash_tag, self.latest_detector, self.latest_text_detector = self.generate_hash(self.label_id, hash_lookup=hash_lookup)
        if self.hash_tag != old_hash_tag:
            self.hash_updated_at = datetime.utcnow()
        self.promote_timestamps()
        session.flush()

    @classmethod
    def update_label_hash(cls, label_id, hash_lookup=None):
        lh = cls.query.filter_by(label_id=label_id).first()
        if lh is None:
            lh = cls(label_id=label_id)
        lh.update(hash_lookup=hash_lookup)

    @classmethod
    def update_all_hash_tags(cls):
        ''' update hash-tag and timestamps for all modified labels'''
        hash_lookup = {}
        for label in Label.query:
            # the user initiated label has a special case where if we add new rotating contents we want to recalculate on everything with the updated list
            # hence we include it in the list even though it will not have any children
            if len(label.weighted_keywords) or len(label.weighted_labels) or len(label.weighted_clf_targets) or cls.get(label.id) or label.name == 'Rotating Content':
                cls.update_label_hash(label.id, hash_lookup=hash_lookup)

    @staticmethod
    def _newer_timestamp(timestamp1, timestamp2):
        if timestamp1 is None:
            return timestamp2
        if timestamp2 is None:
            return timestamp1
        return max(timestamp1, timestamp2)

    @classmethod
    def _get_detector_hash(cls, label_id):
        latest_detector = None
        latest_text_detector = None
        hash_str = ""

        query = session.query(WeightedClfTarget.clf_target_id,
                              WeightedClfTarget.weight,
                              AbstractClassifier.enabled_since,
                              AbstractClassifier.updated_at)
        query = query.filter_by(label_id=label_id)
        query = query.join(WeightedClfTarget.clf_target)
        query = query.join(ClassifierTarget.clf)
        query = query.filter(AbstractClassifier.enabled_since != None)
        query = query.order_by(ClassifierTarget.id)
        for clf_target_id, wt, enabled_since, updated_at in query:
            clf_target = ClassifierTarget.get(clf_target_id)
            hash_str += "%s%s" % (clf_target_id, wt)
            if isinstance(clf_target.clf, AbstractDetector):
                latest_detector = cls._newer_timestamp(latest_detector,
                                                       enabled_since)
            elif isinstance(clf_target.clf, AbstractTextDetector):
                detector_timestamp = max(enabled_since, updated_at)
                latest_text_detector = cls._newer_timestamp(
                    latest_text_detector, detector_timestamp)

        return hash_str, latest_detector, latest_text_detector

    @classmethod
    def generate_hash(cls, label_id, hash_lookup=None):
        """Generates the hash tag for the given label by iterating
        its weighted labels, weighted keywords, weighted detectors,
        domain name detectors and weighted text_detectors recursively
        """
        hash_lookup = hash_lookup or {}
        if label_id not in hash_lookup:
            label = Label.get(label_id)
            assert label is not None, "label %s Does not exist" % label_id

            hash_str = ""
            latest_detector = None
            latest_text_detector = None

            if label.name == 'Rotating Content':
                max_id = session.query(func.max(RotatingContentPage.id)).scalar()
                hash_str = str(max_id)
            else:
                query = session.query(WeightedLabel.child_id, WeightedLabel.weight)
                query = query.filter_by(parent_id=label_id).order_by(WeightedLabel.child_id)
                for child_id, weight in query:
                    child_hash, child_latest_detector, child_latest_text_detector = cls.generate_hash(child_id, hash_lookup=hash_lookup)
                    hash_str += "%s%s" % (child_hash, weight)
                    latest_detector = cls._newer_timestamp(latest_detector, child_latest_detector)
                    latest_text_detector = cls._newer_timestamp(latest_text_detector, child_latest_text_detector)

                query = session.query(Keyword.text, WeightedKeyword.body_weight, WeightedKeyword.title_weight)
                query = query.filter(Keyword.id==WeightedKeyword.keyword_id, WeightedKeyword.label_id==label_id).order_by(Keyword.id)
                for text, bw, tw in query:
                    hash_str += "%s%s%s" % (text.encode('utf-8'), bw, tw)

                det_hash_str, latest_det_ts, latest_text_det_ts = cls._get_detector_hash(label_id)
                hash_str += det_hash_str
                latest_detector = cls._newer_timestamp(
                    latest_detector, latest_det_ts)
                latest_text_detector = cls._newer_timestamp(
                    latest_text_detector, latest_text_det_ts)

                query = session.query(DomainNameDetector.domain_name, DomainNameDetector.weight)
                query = query.filter_by(target_label_id=label_id).order_by(DomainNameDetector.id)
                for domain_name, wt in query:
                    hash_str += "%s%s" % (domain_name, wt)

                hash_str += "%s" % label.decision_threshold

            hash_tag = hashlib.sha1(hash_str).hexdigest()
            hash_lookup[label_id] = hash_tag, latest_detector, latest_text_detector
        return hash_lookup[label_id]
