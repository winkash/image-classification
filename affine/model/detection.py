import logging
import os, time
from datetime import datetime
from collections import defaultdict

from sqlalchemy.orm import validates
from sqlalchemy.schema import ForeignKeyConstraint
from sqlalchemy.ext.hybrid import hybrid_property
from vendor import langid

from affine import config
from affine.aws import sqs, s3client
from affine.model.base import *
from affine.model.detector_logging import detector_log
from affine.model._sqla_imports import *
from affine.retries import retry_operation
from affine import librato_tools

__all__ = [
    'ResultAggregator',
    # Abstract class for all classifiers
    'AbstractClassifier',
    # Vision detectors
    'AbstractDetector', 'AbstractBetaDetector', 'MotionHistDetector',
    'SceneDetector', 'SpatialSceneDetector', 'FaceDetectClassifier',
    'FaceRecognizeClassifier', 'VideoMotionColorDetector',
    'TextDetectClassifier', 'LogoClassifier', 'StaticVideoClassifier', 'CnnClassifier',
    'TextRecognizeClassifier',
    # Text detectors
    'AbstractTextDetector', 'TopicModelDetector', 'LanguageDetector',
    'DomainNameDetector', 'LdaDetector', 'NerDetector', 'SentimentClassifier',
    # Detection results
    'VideoDetectorResult', 'ImageDetectorResult', 'BoxDetectorResult',
    'TextDetectorResult', 'BetaVideoDetection', 'TextBoxResult',
    # LDA model
    'LdaModel',
    # URL classifier
    'UrlClassifier',
    # NamedEntity classifier
    'NamedEntityClassifier',
    # misc
    'ignore_beta_detectors'
]

logger = logging.getLogger(__name__)

def time_process_video(func):
    """Decorator to time a process_video and submit duration to Librato.

    Returns:
        a decorator function
    """
    metric = 'classification.classifiers.process_video.duration'
    def decorator(clf, *args, **kwargs):
        t1 = time.time()
        ret = func(clf, *args, **kwargs)
        t2 = time.time()
        duration_in_seconds = t2 - t1
        if clf.SUBMIT_TO_LIBRATO:
            librato_tools.submit_value(metric=metric, value=duration_in_seconds,
                pid_suffix=False, source=str(clf.id))
        return ret
    return decorator


class ResultAggregator(object):

    """ This class is used to aggregate all detector results
    on a video. Ususally, the object's result_dict is returned
    as the detector's judge_video method.
    """

    def __init__(self):
        self._results = {}

    def add_image_result(self, timestamp, label_id):
        if 'image_results' not in self._results:
            self._results['image_results'] = defaultdict(list)
        self._results['image_results'][timestamp].append(label_id)

    def add_box_result(self, box_id, label_id):
        if 'box_results' not in self._results:
            self._results['box_results'] = defaultdict(list)
        self._results['box_results'][box_id].append(label_id)

    def add_new_box(self, x, y, w, h, time, box_type, label_id=None):
        if 'new_boxes' not in self._results:
            self._results['new_boxes'] = []
        self._results['new_boxes'].append(
            (x, y, w, h, time, box_type, label_id))

    def add_video_result(self, label_id):
        if 'video_results' not in self._results:
            self._results['video_results'] = []
        self._results['video_results'].append(label_id)

    def add_face_info(self, box_id, conf, parts):
        if 'face_infos' not in self._results:
            self._results['face_infos'] = {}
        self._results['face_infos'][box_id] = (conf, parts)

    @property
    def result_dict(self):
        return self._results


class ModelGrabberMixin(object):

    """ Methods to download models and get local paths """

    def grab_files(self):
        bucket = config.s3_detector_bucket()
        retry_operation(s3client.download_tarball,
            bucket, self.tarball_basename, self.local_dir(), sleep_time=0.1,
            error_class=IOError)

    def local_path(self, filename, check=True):
        path = os.path.join(self.local_dir(), filename)
        if check:
            assert os.path.exists(path), path
        return path

    def local_dir(self):
        destination = self.tarball_basename
        if hasattr(self, 'updated_at'):
            date_str = self.updated_at.strftime('%Y-%m-%d-%H-%M-%S')
            destination += '_' + date_str
        return os.path.join(config.scratch_detector_path(), destination)

    @property
    def tarball_basename(self):
        return '%s_%d' % (self._cls, self.id)


class AbstractClassifier(ModelGrabberMixin, Base):

    """ Class that abstracts the various classifier types.
    Provides the generic interface for running classifiers and saving results.
    """
    __tablename__ = "abstract_classifiers"

    id = Column('id', Integer, primary_key=True)
    _uuid = Column('uuid', CHAR(length=36), default=new_guid)
    name = Column(Unicode(128), nullable=False)
    enabled_since = Column(DateTime)
    updated_at = Column(Timestamp, server_default=func.now())

    _cls = Column('cls', String(50), nullable=False)
    __mapper_args__ = dict(polymorphic_on=_cls)

    __table_args__ = (UniqueConstraint('cls', 'name',
                                       name='abstract_classifiers_cls_name'),
                      {})

    def check_criteria_to_enable(self):
        """Should be overwritten by subclasses to add any criteria that must
        be satisfied by classifier before it is enabled
        """
        pass

    def enable(self):
        """Enable this classifier so that it can run
        as part of the classification pipeline on videos/pages"""
        self.check_criteria_to_enable()
        if self.enabled_since is None:
            self.enabled_since = datetime.utcnow()
        session.flush()

    def disable(self):
        self.enabled_since = None
        session.flush()

    def add_targets(self, labels):
        from affine.model.classifier_target_labels import ClassifierTarget
        for l in labels:
            ClassifierTarget.get_or_create(self, l)
        session.refresh(self)

    # caveat: statements like AbstractDetector.query.update({'enabled_since': ...})
    # are not validated
    @validates('enabled_since')
    def _validate_enabled_since(self, _, value):
        error_msg = "'enabled_since' must be 'None' for beta detectors"
        assert not isinstance(
            self, AbstractBetaDetector) or value is None, error_msg
        return value

    @property
    def clf_target(self):
        from affine.model.classifier_target_labels import ClassifierTarget
        return ClassifierTarget.query.filter_by(clf_id=self.id).one()

    def get_clf_target(self, target_label_id):
        ''' Returns ClassifierTarget instance for the detector and target label'''
        from affine.model.classifier_target_labels import ClassifierTarget
        return ClassifierTarget.query.filter_by(clf_id=self.id,
                                                target_label_id=target_label_id).one()


class AbstractTextDetector(AbstractClassifier):

    __tablename__ = "text_detectors"
    __mapper_args__ = dict(polymorphic_identity='text_detector')

    id = Column('id', Integer, primary_key=True)

    __table_args__ = (
        ForeignKeyConstraint(['id'], ['abstract_classifiers.id']),
    )

    def get_coverage(self):
        """Returns the fraction of web pages the detector has run on.
        The detector must be enabled.
        """
        from affine.model.web_pages import WebPage
        msg = 'Detector is not enabled'
        assert self.enabled_since, msg
        last_update = max(self.enabled_since, self.updated_at)
        query = session.query(func.count(WebPage.id.distinct())).filter(
                WebPage.domain != 'set.tv')
        query = query.join(WebPage.inventory)
        query = query.filter(WebPage.text_detection_update != None)
        total_num_pages = query.scalar()
        query = query.filter(WebPage.text_detection_update > last_update)
        num_pages = query.scalar()
        return num_pages / float(total_num_pages)

    @classmethod
    def delete_detector_results(cls, page, detector_ids):
        """Delete tdrs for a given page for a given set of detectors"""
        from affine.model.classifier_target_labels import ClassifierTarget
        if detector_ids:
            query = session.query(ClassifierTarget.id).filter(
                ClassifierTarget.clf_id.in_(detector_ids))
            clf_target_ids = {idx for (idx,) in query}
            TextDetectorResult.query.filter_by(page=page).filter(
                TextDetectorResult.clf_target_id.in_(clf_target_ids)).\
                delete(synchronize_session=False)

    def save_result(self, page_id, target_label_id=None):
        if not target_label_id:
            clf_target = self.clf_target
        else:
            clf_target = self.get_clf_target(target_label_id)
        TextDetectorResult.log_result(page_id, clf_target.id)


class TopicModelDetector(AbstractTextDetector):
    __mapper_args__ = dict(polymorphic_identity='topic_model')


class LdaModel(Base, ModelGrabberMixin):
    __tablename__ = "lda_models"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, server_default=func.now())

    @property
    def tarball_basename(self):
        return '%s_%d' % ("lda_model", self.id)


class LdaDetector(AbstractTextDetector):
    __tablename__ = "lda_detectors"
    __mapper_args__ = dict(polymorphic_identity='lda_detector')

    id = Column(
        'id', Integer, ForeignKey('text_detectors.id'), primary_key=True)
    lda_model_id = Column(Integer, ForeignKey('lda_models.id'), nullable=False)

    def check_criteria_to_enable(self):
        assert self.lda_model_id is not None,\
            'Cannot enable an LdaDetector if it does not have an Lda Model'


class NerDetector(AbstractTextDetector):
    __mapper_args__ = dict(polymorphic_identity='ner_detector')
    SVM_MODEL = 'model.svm'


class UrlClassifier(AbstractTextDetector):
    __mapper_args__ = dict(polymorphic_identity='url_classifier')


class NamedEntityClassifier(AbstractTextDetector):
    __mapper_args__ = dict(polymorphic_identity='named_entity_classifier')

class SentimentClassifier(AbstractTextDetector):
    __mapper_args__ = dict(polymorphic_identity='sentiment_classifier')


class DomainNameDetector(Base):
    __tablename__ = "domain_detectors"

    id = Column(Integer, primary_key=True)
    domain_name = Column(
        VARCHAR(128, charset='ascii', collation='ascii_general_ci'),
        nullable=False)
    target_label_id = Column(Integer, ForeignKey('labels.id'), nullable=False)
    weight = Column(Integer, nullable=False, default=100, server_default='100')
    updated_at = Column(DateTime, server_default=func.now())

    target_label = relation('Label', backref="domain_detectors")

    @classmethod
    def get_or_create(cls, label, domain_name, weight=100):
        if not domain_name.startswith("."):
            domain_name = "." + domain_name
        dnd = cls.query.filter_by(
            target_label=label, domain_name=domain_name).first()
        if dnd is None:
            dnd = cls(domain_name=domain_name, target_label=label)
        dnd.weight = weight
        session.flush()
        return dnd


class LanguageDetector(AbstractTextDetector):
    __mapper_args__ = dict(polymorphic_identity='language_detector')
    code_lang_map = {'en': 'English', 'es': 'Spanish', 'zh': 'Chinese',
                     'fr': 'French', 'de': 'German', 'ru': 'Russian',
                     'it': 'Italian', 'pt': 'Portuguese', 'ar': 'Arabic',
                     'vi': 'Vietnamese', 'ko': 'Korean'}

    @classmethod
    def detect_language(cls, text):
        ''' Returns the language label name
            given a text string as input
        '''
        lang_code, confidence = langid.classify(text)
        if confidence < 0.8:
            return "Unknown Language"
        return cls.code_lang_map.get(lang_code, "Foreign Language")


class TextDetectorResult(Base):
    __tablename__ = "text_detector_results"
    page_id = Column(
        Integer, ForeignKey('web_pages.id'),
        nullable=False, primary_key=True)
    clf_target_id = Column(Integer, ForeignKey('classifier_targets.id'),
                           primary_key=True, nullable=False)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    clf_target = relation('ClassifierTarget',
                          backref=backref('text_results',
                                          passive_deletes=True))
    page = relation('WebPage', backref=backref(
        'text_detector_results', passive_deletes=True))

    def __unicode__(self):
        return u'<text detection (%s) (%s)>' \
            % (self.clf_target.name, self.page.id)

    @classmethod
    def set_result(cls, page_id, clf_target_id):
        kwargs = {'page_id': page_id, 'clf_target_id': clf_target_id}
        if cls.query.filter_by(**kwargs).first() is None:
            cls(**kwargs)
        session.flush()

    @classmethod
    def log_result(cls, page_id, clf_target_id):
        detector_log("TDR", page_id, clf_target_id)

    @classmethod
    def load_from_file(cls, tdr_file, on_duplicate='ignore'):
        cols = 'page_id, clf_target_id'
        cls._load_from_file(tdr_file, cols, on_duplicate)


class AbstractDetector(AbstractClassifier):

    """ Class that abstracts the various vision detectors types.
    Provides a generic interface for running detectors and saving results.
    """
    __tablename__ = "abstract_detectors"
    __mapper_args__ = dict(polymorphic_identity='abstract_detector')


    id = Column('id', Integer, primary_key=True)
    run_group = 0

    __table_args__ = (
        ForeignKeyConstraint(['id'], ['abstract_classifiers.id']),
    )
    SUBMIT_TO_LIBRATO = True

    def judge_video(self, video_path=None, imagedir=None):
        """Run classifier on given video and return all detection results.
        detection_results is a dict whose schema looks like following

        detection_results = {
            'image_results' : { timestamp : [list_of_label_ids], ...},
            'box_results'   : { box_id : [list of label_ids ...] },
            'video_results' : [list of label_ids]
        }

        """
        return {}

    @time_process_video
    def process_video(self, video_id, video_path=None, imagedir=None):
        """Run detection on a given video and save the results to the DB."""
        detection_results = self.judge_video(video_path=video_path,
                                             imagedir=imagedir)
        self.save_results(video_id, detection_results)

    def save_results(self, video_id, detection_results):
        from affine.model.videos import Video
        from affine.model.boxes import Box
        clf_targets = {
            ct.target_label_id: ct.id for ct in self.clf_targets
        }

        image_results = detection_results.get('image_results', {})
        for timestamp, label_ids in image_results.iteritems():
            for l in label_ids:
                self.save_image_result(video_id, timestamp, clf_targets[l])

        label_ids = detection_results.get('video_results', [])
        for l in label_ids:
            self.save_video_result(video_id, clf_targets[l])

        box_results = detection_results.get('box_results', {})
        for box_id, label_ids in box_results.iteritems():
            for l in label_ids:
                self.save_box_result(box_id, clf_targets[l])

        new_boxes = detection_results.get('new_boxes', [])
        video = Video.get(video_id)
        for (x, y, w, h, time, box_type, l) in new_boxes:
            box_id = Box.get_or_create(x, y, w, h, video, time, box_type)
            if l:
                self.save_box_result(box_id, clf_targets[l])

        face_infos = detection_results.get('face_infos', {})
        for box_id, (conf, parts) in face_infos.iteritems():
            self.save_face_info(box_id, conf, parts)
        session.flush()

    def save_box_result(self, box_id, clf_target_id):
        BoxDetectorResult.log_result(box_id, clf_target_id)

    def save_image_result(self, video_id, timestamp, clf_target_id):
        ImageDetectorResult.log_result(video_id, timestamp, clf_target_id)

    def save_video_result(self, video_id, clf_target_id):
        """Store the result of running this detector on a video."""
        VideoDetectorResult.log_result(video_id, clf_target_id)

    def save_face_info(self, box_id, conf, parts):
        from affine.model.boxes import FaceInfo
        fi = FaceInfo.get(box_id)
        if not fi:
            FaceInfo(box_id=box_id, confidence=conf, parts=parts)
        else:
            fi.confidence = conf
            fi.parts = parts

    def _process_results(self, image_results, video_result):
        ra = ResultAggregator()
        for timestamp, res in image_results.iteritems():
            if res:
                ra.add_image_result(timestamp, self.clf_target.target_label_id)
        if video_result:
            ra.add_video_result(self.clf_target.target_label_id)

        return ra.result_dict


def ignore_beta_detectors(query):
    '''Takes a query on AbstractDetector and filters out beta detectors'''
    abd = AbstractBetaDetector
    return query.outerjoin(abd,
                           abd.id == AbstractDetector.id).filter(abd.id == None)


class AbstractBetaDetector(AbstractDetector):

    """Class that abstracts out various beta detectors."""

    __tablename__ = "abstract_beta_detectors"
    __mapper_args__ = dict(polymorphic_identity='abstract_beta_detector')
    id = Column('id', Integer, ForeignKey('abstract_detectors.id'),
                 primary_key=True)
    active_since = Column(DateTime)
    _QUEUE_NAME_FMT = '%(env)s-beta-%(id)s'
    SUBMIT_TO_LIBRATO = False

    @classmethod
    def get(cls, detector_id):
        return cls.query.filter_by(id=detector_id).first()

    def save_results(self, video_id, *args):
        super(AbstractBetaDetector, self).save_results(video_id, *args)
        BetaVideoDetection.record(self.id, video_id)

    def activate(self):
        if self.active_since is None:
            self.active_since = datetime.utcnow()
        session.flush()

    def deactivate(self):
        self.active_since = None
        session.flush()

    @property
    def queue(self):
        if not hasattr(self, '_queue'):
            assert self.id is not None, 'need to persist the detector'
            self._queue = retry_operation(sqs.create_queue, self._queue_name)
        return self._queue

    @property
    def _queue_name(self):
        return self._QUEUE_NAME_FMT % dict(env=config.get('env'), id=self.id)

    def write_to_queue(self, video_id):
        sqs.write_to_queue(self.queue, dict(video_id=video_id))

    def read_from_queue(self):
        msg = sqs.read_from_queue(self.queue)
        if msg is None:
            return
        sqs.delete_message(msg)
        msg_dict = msg.get_body()
        video_id = msg_dict['video_id']
        return video_id


class MotionHistDetector(AbstractDetector):

    """Motion histogram SVM scene detector"""
    __tablename__ = "motion_detectors"
    __mapper_args__ = dict(polymorphic_identity='motion_detector')

    id = Column('id', Integer, ForeignKey('abstract_detectors.id'),
                 primary_key=True)
    threshold = Column(Float, nullable=False)
    added = Column(DateTime, default=datetime.utcnow)
    SUBMIT_TO_LIBRATO = False

    def judge_video(self, video_path=None, imagedir=None):
        from affine.detection.vision import motion
        svm_path = self.local_path('motion.svm')
        score = motion.rate_video(video_path, svm_path)
        video_result = score >= self.threshold
        image_results = {}
        return self._process_results(image_results, video_result)


class SpatialSceneDetector(AbstractDetector):

    """ Spatial scene detector """
    __tablename__ = "spatial_scene_detector"
    __mapper_args__ = dict(polymorphic_identity='spatial_scene_detector')

    id = Column('id', Integer, ForeignKey('abstract_detectors.id'),
                 primary_key=True)
    video_threshold = Column(Integer, nullable=False, default=2)
    image_threshold = Column(Float)
    SUBMIT_TO_LIBRATO = False

    def judge_video(self, video_path=None, imagedir=None):
        from affine.detection.vision.spatial_scene\
            import detection as spatial_scene_detection
        res = spatial_scene_detection.judge_video(
            imagedir, self.local_dir(), self.video_threshold,
            self.image_threshold)
        return self._process_results(*res)


class SceneDetector(AbstractDetector):

    """ Scene detector"""
    __tablename__ = "scene_detector"
    __mapper_args__ = dict(polymorphic_identity='scene_detector')

    id = Column('id', Integer, ForeignKey('abstract_detectors.id'),
                 primary_key=True)
    feature_type = Column(
        Enum("SURF", "SURFEX", strict=True), nullable=False, default='SURFEX')
    pca_dimensions = Column(Integer, nullable=False)
    image_threshold = Column(Float, nullable=False, default=0.0)
    video_threshold = Column(Integer, nullable=False, default=2)
    added = Column(DateTime, default=datetime.utcnow)
    SUBMIT_TO_LIBRATO = False

    def judge_video(self, video_path=None, imagedir=None):
        from affine.detection.vision.scene import detection as scene_detection
        res = scene_detection.judge_video(imagedir,
                                          self.local_dir(),
                                          self.image_threshold,
                                          self.video_threshold)
        return self._process_results(*res)


class BetaVideoDetection(Base):
    __tablename__ = "beta_video_detections"
    video_id = Column(Integer, ForeignKey('videos.id'), primary_key=True)
    beta_detector_id = Column(CHAR(length=36), ForeignKey('abstract_beta_detectors.id'),
                              primary_key=True)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    beta_detector = relation('AbstractBetaDetector',
                             backref=backref('beta_video_detections', passive_deletes=True))

    video = relation('Video',
                     backref=backref('beta_video_detections', passive_deletes=True))

    def __unicode__(self):
        return u'<beta video detection (%s) (%s)>'\
            % (self.beta_detector.name, self.video.id)

    @classmethod
    def record(cls, beta_detector_id, video_id):
        kwargs = dict(video_id=video_id, beta_detector_id=beta_detector_id)
        if cls.query.filter_by(**kwargs).first() is None:
            cls(**kwargs)
        session.flush()


class VideoDetectorResult(Base):
    __tablename__ = "video_detector_results"
    video_id = Column(
        Integer, ForeignKey('videos.id'), nullable=False, primary_key=True)
    clf_target_id = Column(Integer, ForeignKey('classifier_targets.id'),
                           primary_key=True, nullable=False)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    clf_target = relation('ClassifierTarget',
                          backref=backref('video_results',
                                          passive_deletes=True))
    video = relation(
        'Video', backref=backref('video_detector_results',
                                 passive_deletes=True))

    def __unicode__(self):
        return u'<video detection (%s) (%s)>' \
            % (self.clf_target.name, self.video.id)

    @classmethod
    def set_result(cls, video_id, clf_target_id):
        kwargs = {'video_id': video_id, 'clf_target_id': clf_target_id}
        if cls.query.filter_by(**kwargs).first() is None:
            cls(**kwargs)
        session.flush()

    @classmethod
    def log_result(cls, video_id, clf_target_id):
        detector_log("VDR", video_id, clf_target_id)

    @classmethod
    def load_from_file(cls, vdr_file, on_duplicate='ignore'):
        cols = 'video_id, clf_target_id'
        cls._load_from_file(vdr_file, cols, on_duplicate)


class BoxDetectorResult(Base):
    __tablename__ = "box_detector_results"
    box_id = Column(
        Integer, ForeignKey('boxes.id'), nullable=False, primary_key=True)
    clf_target_id = Column(Integer, ForeignKey('classifier_targets.id'),
                           primary_key=True, nullable=False)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    clf_target = relation('ClassifierTarget',
                          backref=backref('box_results',
                                          passive_deletes=True))
    box = relation(
        'Box', backref=backref('detector_results', passive_deletes=True))

    __table_args__ = (UniqueConstraint(
        'box_id', 'clf_target_id',
        name='box_detector_results_box_id_clf_target_id'), {})

    @classmethod
    def log_result(cls, box_id, clf_target_id):
        detector_log("BDR", box_id, clf_target_id)

    @classmethod
    def load_from_file(cls, bdr_file, on_duplicate='ignore'):
        cols = 'box_id, clf_target_id'
        cls._load_from_file(bdr_file, cols, on_duplicate)

    def __unicode__(self):
        return u'<Box detection (%s) (%s)>' % (self.clf_target.name, self.box_id)


class ImageDetectorResult(Base):
    __tablename__ = "image_detector_results"
    video_id = Column(
        Integer, ForeignKey('videos.id'), nullable=False, primary_key=True)
    clf_target_id = Column(Integer, ForeignKey('classifier_targets.id'),
                           primary_key=True, nullable=False)
    time = Column(Integer, nullable=False, primary_key=True)
    timestamp = Column(Timestamp, nullable=False)

    video = relation(
        'Video', backref=backref('image_detector_results',
                                 passive_deletes=True))
    clf_target = relation('ClassifierTarget',
                          backref=backref('image_results',
                                          passive_deletes=True))

    __table_args__ = (UniqueConstraint(
        'video_id', 'time', 'clf_target_id',
        name='image_detector_results_video_id_time_clf_target_id'), {})

    def __unicode__(self):
        return u'ImageDetectorResult <video_id:%s, time:%s, clf_target:%s>' \
            % (self.video_id, self.time, self.clf_target.name)

    @classmethod
    def set_result(cls, video_id, time, clf_target_id, timestamp=None):
        kwargs = {
            'video_id': video_id, 'time': time, 'clf_target_id': clf_target_id}
        idr = cls.query.filter_by(**kwargs).first()
        if idr is None:
            idr = cls(**kwargs)
        if timestamp is not None:
            idr.timestamp = timestamp
        session.flush()

    @classmethod
    def log_result(cls, video_id, time, clf_target_id):
        detector_log("IDR", clf_target_id, video_id, time)

    @classmethod
    def load_from_file(cls, idr_file, on_duplicate='ignore'):
        cols = 'clf_target_id, video_id, time'
        cls._load_from_file(idr_file, cols, on_duplicate)


class FaceRecognizeClassifier(AbstractDetector):

    __mapper_args__ = dict(polymorphic_identity='face_recognize_classifier')
    SUBMIT_TO_LIBRATO = False
    run_group = 1

    @time_process_video
    def process_video(self, video_id, video_path=None, imagedir=None):
        from affine.detection.vision.facerec.detection import recognize_judge_video
        res = recognize_judge_video(self.local_dir(), video_id, imagedir)
        self.save_results(video_id, res)


class FaceDetectClassifier(AbstractDetector):

    __mapper_args__ = dict(polymorphic_identity='face_detect_classifier')
    SUBMIT_TO_LIBRATO = False

    def judge_video(self, video_path=None, imagedir=None):
        from affine.detection.vision.facerec.detection import detect_judge_video
        return detect_judge_video(imagedir)

    def grab_files(self):
        # TODO this classifier has no model files. For the time being
        # leave this as a no-op so wrangler doesn't complain.
        pass


class VideoMotionColorDetector(AbstractDetector):

    """ Motion-Color based detector"""
    __tablename__ = "motion_color_detector"
    __mapper_args__ = dict(polymorphic_identity='motion_color_detector')

    id = Column('id', Integer, ForeignKey('abstract_detectors.id'),
                 primary_key=True)
    confidence_th = Column(Float, nullable=False, default=0.65)
    acceptance_th = Column(Float, nullable=False, default=0.7)
    added = Column(DateTime, default=datetime.utcnow)

    def process_video(self, video_id, video_path=None, imagedir=None):
        """Run detection on a given video and save the results to the DB.
            NOTE: This method overrides the abstract class one
            to pass more information to self.judge_video
        """
        res = self.judge_video(video_id, video_path, imagedir)
        self.save_results(video_id, res)

    def judge_video(self, video_id, video_path, imagedir=None):
        from affine.detection.vision.video_motioncolor\
            import detection as video_mc_detection
        res = video_mc_detection.judge_video(video_id=video_id,
                                             video_path=video_path,
                                             model_dir=self.local_dir(),
                                             conf_th=self.confidence_th,
                                             accept_th=self.acceptance_th)
        return self._process_results(*res)


class LogoClassifierMixin(object):

    def judge_video(self, video_path=None, imagedir=None):
        from affine.detection.vision.logo_recognition.detection import judge_video
        return judge_video(self.local_dir(), imagedir)


class LogoClassifier(LogoClassifierMixin, AbstractDetector):
    __mapper_args__ = dict(polymorphic_identity="logo_classifier")


class TextDetectClassifier(AbstractDetector):

    """
    In-Image Text detector in the wild
    """
    __tablename__ = "text_detect_classifier"
    __mapper_args__ = dict(polymorphic_identity='text_detect_classifier')

    id = Column('id', Integer, ForeignKey('abstract_detectors.id'),
                 primary_key=True)
    pred_thresh = Column(Float, nullable=False, default=0.7)
    added = Column(DateTime, default=datetime.utcnow)

    def _process_results(self, bounding_rects):
        ra = ResultAggregator()
        for time in bounding_rects.keys():
            for h, w, x, y in bounding_rects[time]:
                ra.add_new_box(x, y, w, h, time,
                               'Text', label_id=self.clf_target.target_label_id)

        if bounding_rects:
            ra.add_video_result(self.clf_target.target_label_id)

        return ra.result_dict

    def judge_video(self, video_path=None, imagedir=None):
        from affine.detection.vision.vision_text_detect \
            import detection as vision_text_detection

        bounding_rects = vision_text_detection.judge_video(
            imagedir=imagedir, model_dir=self.local_dir(),
            word_det_th=self.pred_thresh)

        return self._process_results(bounding_rects)


class TextRecognizeClassifier(AbstractDetector):

    """In-Image Text recognizer in the wild"""
    __tablename__ = "text_recognize_classifier"
    __mapper_args__ = dict(polymorphic_identity='text_recognize_classifier')

    id = Column('id', Integer, ForeignKey('abstract_detectors.id'),
                 primary_key=True)
    pred_thresh = Column(Float, nullable=False, default=0.2)
    added = Column(DateTime, default=datetime.utcnow)
    run_group = 1

    @time_process_video
    def process_video(self, video_id, video_path=None, imagedir=None):
        """Takes in a video id to find text boxes to recognize."""
        words = self.judge_video(video_id, video_path, imagedir)
        self.save_results(words)

    def judge_video(self, video_id, video_path=None, imagedir=None):
        from affine.detection.vision.vision_text_recognize \
            import detection as vision_text_recognize
        return vision_text_recognize.judge_video(
            video_id=video_id, imagedir=imagedir, model_dir=self.local_dir(),
            rec_th=self.pred_thresh)

    def save_results(self, words):
        from affine.detection.vision.vision_text_recognize \
            import detection as vision_text_recognize
        vision_text_recognize.save_results(words)


class TextBoxResult(Base):
    __tablename__ = "text_box_results"
    box_id = Column(
        Integer, ForeignKey('boxes.id'), nullable=False, primary_key=True)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())
    text = Column(CHAR(length=36), nullable=False)

    __table_args__ = (UniqueConstraint(
        'box_id', name='text_box_results_box_id'), {})

    @classmethod
    def log_result(cls, box_id, text):
        detector_log("TBR", box_id, text)

    @classmethod
    def load_from_file(cls, tbr_file, on_duplicate='ignore'):
        cols = 'box_id, text'
        cls._load_from_file(tbr_file, cols, on_duplicate)

    def __unicode__(self):
        return u'<Text recognition (%s) (%s)>' % (self.box_id, self.text)


class StaticVideoClassifier(AbstractDetector):

    """Static Video Classifier"""
    __tablename__ = "static_video_classifier"
    __mapper_args__ = dict(polymorphic_identity='static_video_classifier')

    id = Column('id', Integer, ForeignKey('abstract_detectors.id'),
                 primary_key=True)

    def process_video(self, video_id, video_path=None, imagedir=None):
        """Run detection on a given video and save the results to the DB."""
        res = self.judge_video(video_id, video_path, imagedir)
        self.save_results(video_id, res)

    def judge_video(self, video_id, video_path, imagedir=None):
        from affine.detection.vision.static_video import detection as \
            static_video_detection
        res = static_video_detection.judge_video(
            video_id, video_path, model_dir=self.local_dir())
        return res


class CnnClassifier(AbstractDetector):

    """ Cnn feature based classifier """
    __mapper_args__ = dict(polymorphic_identity='cnn_classifier')

    def judge_video(self, video_path=None, imagedir=None):
        from affine.detection.vision.cnn_classifier \
            import detection as cnn_detection
        res = cnn_detection.judge_video(
            image_dir=imagedir, model_dir=self.local_dir())
        return res
