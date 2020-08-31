from sqlalchemy.orm import validates
from sqlalchemy.ext.hybrid import hybrid_property
from affine.model._sqla_imports import *
from affine.model.base import Base


__all__ = ["ClassifierTarget"]


class ClassifierTarget(Base):
    __tablename__ = "classifier_targets"

    id = Column(Integer, primary_key=True)
    clf_id = Column(
        Integer, ForeignKey('abstract_classifiers.id'), nullable=False)
    target_label_id = Column(Integer, ForeignKey('labels.id'), nullable=False)

    # All MTurk related columns
    video_qa_enabled = Column(Boolean, nullable=False, default=False)
    collage_count = Column(Integer, default=10)
    page_qa_enabled = Column(Boolean, nullable=False, default=False)
    screenshot_count = Column(Integer, default=10)
    image_qa_enabled = Column(Boolean, nullable=False, default=False)
    image_qa_count = Column(Integer, default=10)
    box_qa_enabled = Column(Boolean, nullable=False, default=False)
    box_qa_count = Column(Integer, default=100)

    clf = relation('AbstractClassifier', backref="clf_targets")
    target_label = relation('Label', backref="target_classifiers")

    def __unicode__(self):
        return '<ClassifierTarget clf:%s target_label:%s>' \
            % (self.clf, self.target_label)

    # the name property is primarily reqd only for the gen_qa_numbers
    # for Mturk VDR, TDR, etc.
    @hybrid_property
    def name(self):
        return "%s:%s" % (self.clf.name, self.target_label.name)

    @classmethod
    def get_or_create(cls, classifier, target_label):
        kwargs = dict(
            clf_id=classifier.id, target_label_id=target_label.id)
        inst = cls.query.filter_by(**kwargs).first()
        if not inst:
            inst = cls.create(**kwargs)
        return inst

    @validates('page_qa_enabled')
    def validate_page_qa_enabled(self, key, value):
        from affine.model.mturk.evaluators import WebPageTextEvaluator
        if value:
            dtc = WebPageTextEvaluator.query.filter_by(
                target_label_id=self.target_label_id).first()
            assert dtc,\
                'Cannot enable page-QA, WebPageTextEvaluator does not exist'
            assert self.clf.updated_at,\
                'Classifier needs to have updated_at ts for QA'

        return value
