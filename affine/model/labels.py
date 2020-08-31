from datetime import datetime
import sys

from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import validates
from sqlalchemy import or_, inspect
from sqlalchemy.schema import ForeignKeyConstraint

from affine.model.base import *
from affine.model.classifier_target_labels import ClassifierTarget
from affine.model.campaigns import Campaign
from affine.model.line_items import LineItem
from affine.model.channels import Channel
from affine.model._sqla_imports import *
from affine.model.detection import AbstractTextDetector, AbstractBetaDetector, AbstractDetector
from vendor.sqlalchemy_fsm import FSMField, transition

__all__ = ['Label', 'AppLabel', 'Keyword', 'WeightedLabel', 'AppLabelRelation', 'WeightedKeyword', 'WeightedClfTarget', 'AbstractLabel', 'LABEL_IGNORE_LIST']

LABEL_IGNORE_LIST = ['HD', 'Quality Player', 'Autoplay', 'Viewability: Above The Fold', 'Short (0-4 min.)', 'Medium (4-20 min.)', 'Long (20+ min.)', 'User Initiated', 'Viewable', 'Rotating Content', 'pre-roll', 'non-pre-roll', 'app']

COVERAGE = 0.6


class AbstractLabel(Base):

    __tablename__ = 'abstract_labels'
    id = Column(Integer, nullable=False, autoincrement=True, primary_key=True)
    name = Column(Unicode(128), unique=True, nullable=False)
    infobright = Column(Boolean, nullable=False, default=False)
    is_public = Column(Boolean, nullable=False, default=False)
    children_count = Column(Integer, nullable=False, default=0)
    state = Column(FSMField, default='draft', nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by = Column(String, default='')
    updated_at = Column(Timestamp, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String, default='')
    _cls = Column('cls', String(50), nullable=False)
    __mapper_args__ = dict(polymorphic_on=_cls, with_polymorphic='*')
    __table_args__ = (
        UniqueConstraint('cls', 'name', name='uniq_label_name'),
    )

    @validates('is_public')
    def validate_is_public(self, key, value):
        if self._cls == "label":
            if self.name not in LABEL_IGNORE_LIST:
                # Check if Label has atleast 1 weighted-relation if it is being made public
                if value:
                    weighted_relations = self.weighted_labels + self.weighted_keywords + self.weighted_clf_targets
                    assert weighted_relations, 'Label cannot be made public as it does not have any weighted-relations!'

                # Check if Label is not being used by any campaign if it is being made private
                else:
                    query = session.query(Label.id).join(Channel.line_items).join(LineItem.campaign)
                    pos_label_in_campaign = query.join(Channel.positive_labels).filter(Label.id==self.id, Campaign.start_date<datetime.utcnow(), Campaign.end_date>datetime.utcnow(), AbstractLabel._cls == "label").first()
                    neg_label_in_campaign = query.join(Channel.negative_labels).filter(Label.id==self.id, Campaign.start_date<datetime.utcnow(), Campaign.end_date>datetime.utcnow(), AbstractLabel._cls == "label").first()

                    assert (pos_label_in_campaign or neg_label_in_campaign) is None, 'Label cannot be made private as it is being used in a Campaign!'

        return value

    @property
    def _model(self):
        return inspect(self).mapper.class_

    def descendants(self):
        """All labels below this one in the tree, including this label"""
        label_ids = self.all_descendant_ids([self.id])
        return session.query(self._model).filter(self._model.id.in_(label_ids)).all()

    def ancestors(self):
        """All labels above this one in the tree, including this label"""
        label_ids = self.all_ancestor_ids([self.id])
        return session.query(self._model).filter(self._model.id.in_(label_ids)).all()

    def descendant_models(self):
        query = session.query(self._model).join(self._inherited().child_label)
        return query.filter(self._inherited().parent_id==self.id).all()

    def ancestor_models(self):
        query = session.query(self._model).join(self._inherited().parent_label)
        return query.filter(self._inherited().child_id==self.id).all()

    def remove_label(self, label):
        self.delete(self.id, label.id)
        session.refresh(self)
        session.refresh(label)

    def remove_labels(self, labels):
        for label in labels:
            self.delete(self.id, label.id)
        session.flush()

    def remove_label_ancestors(self, labels):
        for label in labels:
            self.delete(label.id, self.id)
        session.refresh(self)
        session.flush()

    def add_weighted_label(self, label, weight=None):
        assert (self._cls == label._cls), 'Cannot add label of different cls type'
        self._inherited()._create(self.id, label.id, weight)
        session.refresh(self)
        session.refresh(label)

    def add_weighted_labels(self, labels, weight=None):
        for label in labels:
            assert (self._cls == label._cls), 'Cannot add label of different cls type'
            self._inherited()._create(self.id, label.id, weight)
        session.refresh(self)
        session.refresh(label)

    @classmethod
    def _all_ancestor_or_descendant_ids(cls, label_ids, ancestors=True):
        if ancestors:
            column1, column2 = cls._inherited().parent_id, cls._inherited().child_id,
        else: # descendants
            column1, column2 = cls._inherited().child_id, cls._inherited().parent_id

        label_ids_to_process = label_ids
        label_ids = set(label_ids)
        while label_ids_to_process:
            query = session.query(column1.distinct())
            query = query.filter(column2.in_(label_ids_to_process))
            new_label_ids = {label_id for [label_id] in query}
            label_ids_to_process = new_label_ids - label_ids
            label_ids.update(new_label_ids)
        return label_ids

    @classmethod
    def all_ancestor_ids(cls, label_ids):
        return cls._all_ancestor_or_descendant_ids(label_ids)

    @classmethod
    def all_descendant_ids(cls, label_ids):
        return cls._all_ancestor_or_descendant_ids(label_ids, ancestors=False)

    @classmethod
    def delete(cls, parent_id, child_id):
        """Deletes a WeightedLabel/AppLabelRelation from the database"""
        cls._inherited().query.filter_by(parent_id=parent_id, child_id=child_id).delete()


class AppLabel(AbstractLabel):
    __mapper_args__ = dict(polymorphic_identity='app_label')

    @classmethod
    def _inherited(cls):
        return AppLabelRelation


class Label(AbstractLabel):
    __tablename__ = 'labels'
    __mapper_args__ = dict(polymorphic_identity='label')
    id = Column(Integer, nullable=False, autoincrement=True, primary_key=True)
    label_type = Column(Enum('keywords', 'flip', 'target', 'spotlight', 'personality', "brand"), unique=False, nullable=True)
    qa_enabled = Column(Boolean, nullable=False, default=False)
    page_qa_enabled = Column(Boolean, nullable=False, default=False)
    screenshot_count = Column(Integer, nullable=False, default=10)
    non_preroll_qa_count = Column(Integer, nullable=False, default=10)
    collage_count = Column(Integer, nullable=False, default=10)
    confidence = Column(Float)
    decision_threshold = Column(Integer, nullable=False, default=100)
    __table_args__ = (
        ForeignKeyConstraint(['id'], ['abstract_labels.id']),
    )

    @classmethod
    def _inherited(cls):
        return WeightedLabel

    @validates('qa_enabled')
    def validate_qa_enabled(self, key, value):
        from affine.model.mturk.evaluators import VideoCollageEvaluator
        if value:
            dtc = VideoCollageEvaluator.query.filter_by(target_label_id=self.id).first()
            assert dtc, 'Cannot enable QA, VideoCollageEvaluator does not exist'

        return value

    @validates('page_qa_enabled')
    def validate_page_qa_enabled(self, key, value):
        from affine.model.mturk.evaluators import WebPageTextEvaluator
        if value:
            dtc = WebPageTextEvaluator.query.filter_by(target_label_id=self.id).first()
            assert dtc, 'Cannot enable page-QA, WebPageTextEvaluator does not exist'

        return value

    # Transition Conditions.
    def can_submit(self):
        from affine.model.mturk.evaluators import VideoCollageEvaluator, WebPageTextEvaluator
        return ((VideoCollageEvaluator.query.filter_by(target_label_id=self.id).first() or
            WebPageTextEvaluator.query.filter_by(target_label_id=self.id).first()))

    def can_approve(self):
        return self.qa_enabled or self.page_qa_enabled

    @transition(source='draft', target='submitted', conditions=[can_submit])
    def submit(self):
        ''' Changes the label state column '''

    @transition(source='submitted', target='approved', conditions=[can_approve])
    def approve(self):
        ''' Changes the label state column '''

    @transition(source=['submitted', 'approved'], target='rejected')
    def reject(self):
        ''' Changes the label state column '''
        self.is_public = False
        WeightedLabel.query.filter(or_(WeightedLabel.child_id == self.id, WeightedLabel.parent_id == self.id)).delete(synchronize_session = False)

    @transition(source=['rejected', 'submitted'], target='draft')
    def redraft(self):
        ''' Changes the label state column '''

    @hybrid_property
    def display_name(self):
        return self.name.replace('IAB:', '')

    @display_name.expression
    def display_name(cls):
        return func.replace(cls.name, 'IAB:', '')

    @classmethod
    def get_or_create(cls, name, is_public=None, label_type=None):
        query = cls.query.filter(cls.name == name)
        if label_type is not None:
            query.filter(cls.label_type == label_type)

        label = query.first()
        if label is None:
            label = cls(name = name)
            if label_type is not None:
                label.label_type = label_type
        if is_public is not None:
            label.is_public = is_public
        session.flush()
        return label

    def _weighted_clf_targets(self, det_cls):
        q = WeightedClfTarget.query.filter_by(label_id=self.id)
        q = q.join(ClassifierTarget,
                   WeightedClfTarget.clf_target_id == ClassifierTarget.id)
        q = q.join(det_cls, ClassifierTarget.clf_id == det_cls.id)
        return q.all()

    @property
    def weighted_detectors(self):
        return self._weighted_clf_targets(AbstractDetector)

    @property
    def weighted_text_detectors(self):
        return self._weighted_clf_targets(AbstractTextDetector)

    def add_weighted_clf_target(self, clf_target, weight):
        assert isinstance(clf_target, ClassifierTarget)
        WeightedClfTarget._create(self.id, clf_target.id, weight)
        session.refresh(self)
        session.refresh(clf_target)

    def remove_clf_target(self, clf_target):
        assert isinstance(clf_target, ClassifierTarget)
        WeightedClfTarget.delete(self.id, clf_target.id)
        session.refresh(self)
        session.refresh(clf_target)

    def add_weighted_keyword(self, keyword, body_weight, title_weight):
        WeightedKeyword._create(self.id, keyword.id, body_weight, title_weight)
        session.refresh(self)
        session.refresh(keyword)

    def remove_keyword(self, keyword):
        WeightedKeyword.delete(self.id, keyword.id)
        session.refresh(self)
        session.refresh(keyword)

    @property
    def rank(self):
        if self.parent_labels:
            return 1 + max(parent.parent_label.rank for parent in self.parent_labels)
        return 0

    @classmethod
    def by_name(cls, name):
        """Returns a label by its name, excluding labels of type 'keywords'"""
        return cls.query.filter_by(name=name).filter(or_(cls.label_type != 'keywords', cls.label_type==None)).first()

    @classmethod
    def get_vision_labels(cls):
        """ Retuns list of label_ids which have weighted detectors but
        no other weighted relations"""
        query = Label.query.filter(Label.weighted_clf_targets,
                                   ~Label.weighted_keywords.any(),
                                   ~Label.weighted_labels.any())
        label_ids = []
        for label in query:
            if label.weighted_detectors and not label.weighted_text_detectors:
                label_ids.append(label.id)

        return label_ids

    @classmethod
    def by_type(cls, label_type):
        return [label.id for label in cls.query if (label.label_type == label_type)]

    def __unicode__(self):
        string = super(Label, self).__unicode__()
        if self.label_type is not None:
            assert string.endswith('>'), string
            string = string[:-1] + ' (' + self.label_type + ')>'
        return string


class Keyword(Base):
    __tablename__ = 'keywords'
    id = Column(Integer, nullable=False, autoincrement=True, primary_key=True)
    text = Column(
        VARCHAR(200, charset='utf8', convert_unicode=True, collation="utf8_bin"),
        unique=True, nullable=False
    )

    @classmethod
    def get_or_create(cls, text):
        keyword = cls.query.filter_by(text=text).first()
        if not keyword:
            keyword = cls(text=text)
            session.flush()
        return keyword


class LabelRelationMixin:
    @classmethod
    def _create(cls, parent_id, child_id, weight=None):
        wl = cls.query.filter_by(parent_id=parent_id, child_id=child_id).first()
        if wl is None:
            wl = cls(parent_id=parent_id, child_id=child_id)
        if cls == WeightedLabel:
            wl.weight = weight
        session.flush()

    @classmethod
    def delete(cls, parent_id, child_id):
        """Deletes an AppLabelRelation from the database"""
        cls.query.filter_by(parent_id=parent_id, child_id=child_id).delete()


class WeightedLabel(Base, LabelRelationMixin):
    """ This object provides an intermediate step in the relation between parent and children labels.
    It allows the weighting of children labels in combination with detectors for the calculation of WebPageLabelResults"""
    __tablename__ = "label_relations"
    parent_id = Column(Integer, ForeignKey('labels.id'), nullable=False, primary_key=True)
    child_id = Column(Integer, ForeignKey('labels.id'), nullable=False, primary_key=True)
    weight = Column(Integer, nullable=False, default=100, server_default='100')
    parent_label = relation('Label', primaryjoin = (parent_id == Label.id), backref=backref('weighted_labels', cascade='all,delete-orphan'))
    child_label = relation('Label', primaryjoin = (child_id == Label.id), backref=backref('parent_labels', cascade='all,delete-orphan'))

    def __unicode__(self):
        return "%s's child label: %s, with weight %s"%(self.parent_label.name, self.child_label.name, self.weight)


class AppLabelRelation(Base, LabelRelationMixin):
    __tablename__ = 'app_label_relations'
    parent_id = Column(Integer, ForeignKey('abstract_labels.id'), nullable=False, primary_key=True)
    child_id = Column(Integer, ForeignKey('abstract_labels.id'), nullable=False, primary_key=True)
    parent_label = relationship('AppLabel', primaryjoin=(parent_id == AppLabel.id), backref=backref('weighted_labels', cascade='all,delete-orphan'))
    child_label = relationship('AppLabel', primaryjoin=(child_id == AppLabel.id), backref=backref('parent_labels', cascade='all,delete-orphan'))

    def __unicode__(self):
        return "{}'s child label: {}".format(self.parent_label.name, self.child_label.name)


class WeightedKeyword(Base):
    """ This object provides an intermediate step in the relation between Label and Keyword.
    It allows weighting of a keyword for the calculation in WebPageLabelResult of whether the label should be applied to a video."""
    __tablename__ = "label_keywords"
    label_id = Column(Integer, ForeignKey('labels.id', ondelete="cascade"), nullable=False, primary_key = True)
    keyword_id = Column(Integer, ForeignKey('keywords.id'), nullable=False, primary_key = True)
    body_weight = Column(Integer, nullable=False, default=0, server_default='0')
    title_weight = Column(Integer, nullable=False, default=0, server_default='0')
    label = relation('Label', backref=backref('weighted_keywords', cascade='all,delete-orphan'))
    keyword = relation('Keyword', backref=backref('label_associations', cascade='all,delete-orphan'))

    @classmethod
    def _create(cls, label_id, keyword_id, body_weight, title_weight):
        # New WKs should be created through Label.add_weighted_keyword
        # This is a private helper for it
        wk = cls.query.filter_by(label_id=label_id, keyword_id=keyword_id).first()
        if wk is None:
            wk = cls(label_id=label_id, keyword_id=keyword_id)
        wk.body_weight = body_weight
        wk.title_weight = title_weight
        session.flush()

    @classmethod
    def delete(cls, label_id, keyword_id):
        """Deletes a WeightedKeyword from the database"""
        cls.query.filter_by(label_id=label_id, keyword_id=keyword_id).delete()

    def __unicode__(self):
        return "%s's keyword: %s, with body weight %s and title weight %s"%(self.label.name, self.keyword.text, self.body_weight, self.title_weight)


class WeightedClfTarget(Base):
    """ This object provides an intermediate step in the relation between
    Labels and ClassifierTargets. It allows weighting of a clf-target
    for the calculation of whether the label should be applied to the page
    """
    __tablename__ = "weighted_clf_targets"
    label_id = Column(Integer, ForeignKey('labels.id', ondelete="cascade"),
                      nullable=False, primary_key=True)
    clf_target_id = Column(Integer, ForeignKey('classifier_targets.id'),
                           primary_key=True, nullable=False)
    weight = Column(Integer, nullable=False)
    label = relation('Label',
                     backref=backref('weighted_clf_targets',
                                     cascade='all,delete-orphan'))
    clf_target = relation('ClassifierTarget',
                          backref=backref('label_associations',
                                          passive_deletes=True))

    @classmethod
    def _create(cls, label_id, clf_target_id, weight):
        # New Wt_clf_targets should be created through
        # Label.add_weighted_clf_target
        # This is a private helper for it
        error_msg = 'cannot weigh a beta detector under a label'
        clf = ClassifierTarget.get(clf_target_id).clf
        assert AbstractBetaDetector.get(clf.id) is None, error_msg

        wct = cls.query.filter_by(label_id=label_id,
                                  clf_target_id=clf_target_id).first()
        if wct is None:
            if isinstance(clf, AbstractTextDetector):
                msg = 'Insufficient coverage'
                assert clf.get_coverage() >= COVERAGE, msg
            wct = cls(label_id=label_id, clf_target_id=clf_target_id)

        wct.weight = weight
        session.flush()

    @classmethod
    def delete(cls, label_id, clf_target_id):
        """Deletes a WeightedClfTarget from the database"""
        cls.query.filter_by(label_id=label_id,
                            clf_target_id=clf_target_id).delete()

    def __unicode__(self):
        return "%s's clf_target: %s, with weight %s" % (self.label.name,
                                                        self.clf_target.name,
                                                        self.weight)
