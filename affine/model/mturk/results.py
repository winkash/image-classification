from collections import defaultdict
from datetime import datetime, timedelta
from logging import getLogger
from sqlalchemy.sql.expression import case
import affine.aws.elasticache as elasticache
from affine.model._sqla_imports import *
from affine.model.base import Base, session, execute
from affine.model.boxes import Box
from affine.model.detection import AbstractTextDetector,\
    VideoDetectorResult, BoxDetectorResult, TextDetectorResult,\
    ImageDetectorResult, FaceRecognizeClassifier
from affine.model.labels import Label
from affine.model.mturk.evaluators import VideoCollageEvaluator,\
    ClickableBoxEvaluator, WebPageTextEvaluator, ClickableImageEvaluator
from affine.model.mturk.hits import BoxHit, MTurkBox, ImageHit, MTurkImage,\
    VideoHit, PageHit
from affine.model.training_data import TrainingImage, TrainingPage,\
    LabelTrainingPage, TrainingBox
from affine.model.videos import Video
from affine.model.web_pages import WebPage, WebPageInventory, \
    base_query, VideoOnPage
from affine.model.classifier_target_labels import ClassifierTarget
from affine.model.web_page_label_results import WebPageLabelResult
from affine.sphinx import SphinxClient


__all__ = [
    'MTurkWebPageLabelResult', 'MTurkVideoDetectorResult',
    'MTurkWebPageScreenshotResult', 'MTurkBoxDetectorResult',
    'MTurkTextDetectorResult', 'MTurkImageDetectorResult',
    'MTurkVideoResult']


BOXES_TO_QA_PER_VIDEO = 2
IMAGES_TO_QA_PER_VIDEO = 30
logger = getLogger(__name__)


class MTurkQaMixin(object):

    @classmethod
    def _generate_QA_numbers_query(cls, start_date, end_date, hit_type, exclude_table):
        count_bools = lambda expr: cast(func.sum(expr), Integer)
        true_positives = case([(hit_type.result == True, 1)], else_=0)
        total = case([(hit_type.result != None, 1)], else_=0)
        conflicts = case([(hit_type.result == None, 1)], else_=0)
        query = session.query(hit_type.label_id, count_bools(true_positives),
            count_bools(total), count_bools(conflicts))
        query = query.outerjoin(exclude_table, exclude_table.hit_id==hit_type.hit_id)
        query = query.filter(exclude_table.hit_id==None)
        query = query.filter(hit_type.timestamp.between(start_date, end_date))
        query = query.filter(hit_type.outstanding==False)
        query = query.group_by(hit_type.label_id)
        return query

    @classmethod
    def _generate_QA_numbers(cls, start_date, end_date, hit_type, exclude_table):
        return cls._generate_QA_numbers_query(start_date, end_date, hit_type, exclude_table).all()


class MTurkWebPageLabelResult(MTurkQaMixin):

    @classmethod
    def enable_qa(cls, label, collage_count, **kw):
        """ Enable QA for a given label """
        evaluator = VideoCollageEvaluator.get_or_create(label)
        label.qa_enabled = True
        label.collage_count = collage_count
        for key, value in kw.iteritems():
            setattr(evaluator, key, value)
        session.flush()

    @classmethod
    def enabled_labels(cls):
        """returns List of labels for which qa is enabled """
        return [l.id for l in session.query(Label.id).
                filter_by(qa_enabled=True)]

    @classmethod
    def get_ignore_page_ids(cls, label_id):
        """ Returns page ids that were part of training """
        training_ids = {ltp.page_id for ltp in LabelTrainingPage.query.
                        filter_by(label_id=label_id)}
        return training_ids

    @classmethod
    def get_ignore_video_ids(cls, label_id):
        """ Videos that have been QA'd for the given label """
        prev_qad_ids = {vh.video_id for vh in VideoHit.query.
                        filter(VideoHit.label_id == label_id)}
        return prev_qad_ids

    @classmethod
    def results_to_qa_for_label(cls, label_id):
        logger.info("Gathering results for label_id : %s", label_id)
        label_results = []

        ignore_video_ids = cls.get_ignore_video_ids(label_id)
        ignore_page_ids = cls.get_ignore_page_ids(label_id)
        query = session.query(WebPageInventory.video_id, WebPageInventory.page_id)
        query = query.join(WebPageLabelResult, WebPageLabelResult.page_id == WebPageInventory.page_id)
        query = query.distinct(WebPageInventory.video_id).filter(WebPageLabelResult.label_id == label_id)
        if ignore_video_ids:
            query = query.filter(~WebPageInventory.video_id.in_(ignore_video_ids))
        if ignore_page_ids:
            query = query.filter(~WebPageInventory.page_id.in_(ignore_page_ids))

        query = query.filter(WebPageInventory.video_id != 0).group_by(WebPageInventory.video_id)
        vid_page_ids = query.order_by(func.rand()).limit(Label.get(label_id).collage_count).all()

        for video_id, page_id in vid_page_ids:
            label_results.append((label_id, video_id, page_id, True))

        return label_results

    @classmethod
    def results_to_qa(cls):
        label_results = []
        for label_id in cls.enabled_labels():
            label_results += cls.results_to_qa_for_label(label_id)

        return label_results

    @classmethod
    def submit_hits(cls):
        """Submit video labels to MTurk for QA"""
        evaluators = {}
        result_set = cls.results_to_qa()
        num_hits_submitted = 0
        for label_id, video_id, wp_id, expected_result in result_set:
            if label_id not in evaluators:
                evaluators[label_id] = VideoCollageEvaluator.query.filter_by(
                    target_label_id=label_id).one()
            evaluator = evaluators[label_id]

            hit_id = evaluator.create_hit(video_id=video_id)
            vh = VideoHit(hit_id=hit_id, label_id=label_id, video_id=video_id,
                page_id=wp_id)
            num_hits_submitted += 1
            session.flush()
        return VideoHit, num_hits_submitted

    @classmethod
    def generate_QA_numbers(cls, start_date, end_date):
        return cls._generate_QA_numbers(start_date,
                                        end_date,
                                        VideoHit,
                                        MTurkVideoDetectorResult
                                       )


class MTurkWebPageScreenshotResult(MTurkQaMixin):

    @classmethod
    def enable_qa(cls, label, screenshot_count, non_preroll_qa_count, **kw):
        """ Enable QA for a given label """
        evaluator = WebPageTextEvaluator.get_or_create(label)
        label.page_qa_enabled = True
        label.screenshot_count = screenshot_count
        label.non_preroll_qa_count = non_preroll_qa_count
        for key, value in kw.iteritems():
            setattr(evaluator, key, value)
        session.flush()

    @classmethod
    def enabled_labels(cls):
        """returns List of labels for which qa is enabled """
        return [l.id for l in session.query(Label.id).
                filter_by(page_qa_enabled=True)]

    @classmethod
    def get_ignore_page_ids(cls, label_id):
        """ Returns page ids that were part of training or have already
        been QA'd for the given label """
        prev_qad_ids = {ph.page_id for ph in PageHit.query.
                        filter(PageHit.label_id == label_id)}
        training_ids = {ltp.page_id for ltp in LabelTrainingPage.query.
                        filter_by(label_id=label_id)}
        return prev_qad_ids | training_ids

    @classmethod
    def non_preroll_results_to_qa_for_label(cls, label_id, page_ids_to_ignore):
        label_to_qa = Label.get(label_id)
        #Set end date such that no pages ingested same day are QAed to allow for
        #all stages of ingestion to complete
        end_date = datetime.utcnow() - timedelta(days=1)

        query = session.query(WebPageInventory.page_id.distinct())
        query = query.join(WebPage, WebPageInventory.page_id == WebPage.id)
        query = query.filter(WebPage.last_crawled_video <= end_date)
        query = query.outerjoin(VideoOnPage, VideoOnPage.page_id == WebPageInventory.page_id)
        query = query.filter(VideoOnPage.page_id == None)
        if page_ids_to_ignore:
           query = query.filter(~WebPageInventory.page_id.in_(page_ids_to_ignore))
        query = query.join(WebPageLabelResult,
            WebPageLabelResult.page_id == WebPageInventory.page_id)
        query = query.filter(WebPageLabelResult.label_id == label_to_qa.id)
        query = query.order_by(func.rand())
        query = query.limit(label_to_qa.non_preroll_qa_count)

        return [page_id for (page_id,) in query]

    @classmethod
    def preroll_results_to_qa_for_label(cls, label_id, page_ids_to_ignore):
        label_to_qa = Label.get(label_id)

        query = session.query(WebPageInventory.page_id.distinct())
        query = query.outerjoin(VideoOnPage, VideoOnPage.page_id == WebPageInventory.page_id)
        query = query.filter(VideoOnPage.page_id != None)
        if page_ids_to_ignore:
            query = query.filter(~WebPageInventory.page_id.in_(page_ids_to_ignore))
        query = query.join(WebPageLabelResult,
                WebPageLabelResult.page_id == WebPageInventory.page_id)
        query = query.filter(WebPageLabelResult.label_id == label_to_qa.id)
        query = query.order_by(func.rand())
        query = query.limit(label_to_qa.screenshot_count)

        return [page_id for (page_id,) in query]

    @classmethod
    def results_to_qa_for_label(cls, label_id):
        logger.info("Gathering results for label_id : %s", label_id)
        page_ids_to_ignore = cls.get_ignore_page_ids(label_id)

        preroll_page_ids = cls.preroll_results_to_qa_for_label(
            label_id, page_ids_to_ignore)
        non_preroll_page_ids = cls.non_preroll_results_to_qa_for_label(
            label_id, page_ids_to_ignore)

        results_to_qa = set()
        for page_id in preroll_page_ids + non_preroll_page_ids:
            results_to_qa.add((label_id, page_id, True))

        return results_to_qa

    @classmethod
    def results_to_qa(cls):
        label_results = []
        for label_id in cls.enabled_labels():
            label_results += cls.results_to_qa_for_label(label_id)
        return label_results

    @classmethod
    def submit_hits(cls):
        """Submit video labels to MTurk for QA"""
        evaluators = {}
        result_set = cls.results_to_qa()
        num_hits_submitted = 0
        for label_id, wp_id, expected_result in result_set:
            if label_id not in evaluators:
                evaluators[label_id] = WebPageTextEvaluator.query.filter_by(
                    target_label_id=label_id).one()
            evaluator = evaluators[label_id]

            hit_id = evaluator.create_hit(page_id=wp_id)
            ph = PageHit(hit_id=hit_id, label_id=label_id, page_id=wp_id)
            num_hits_submitted += 1
            session.flush()
        return PageHit, num_hits_submitted

    @classmethod
    def generate_QA_numbers(cls, start_date, end_date, non_prerolls_only=False):
        if not non_prerolls_only:
            return cls._generate_QA_numbers(start_date, end_date, PageHit, MTurkTextDetectorResult)

        return cls._generate_QA_numbers_query(start_date, end_date, PageHit, MTurkTextDetectorResult).outerjoin(
                    VideoOnPage, VideoOnPage.page_id == PageHit.page_id).filter(
                    VideoOnPage.page_id==None).all()


class MTurkDetectorResultMixin(MTurkQaMixin):

    @classmethod
    def enable_qa(cls, clf_target, qa_count, **kw):
        assert isinstance(clf_target, ClassifierTarget), \
            "Can only enable ClassifierTargets, got %s" % clf_target
        evaluator = cls.EVALUATOR_CLS.get_or_create(clf_target.target_label)
        for key, value in kw.iteritems():
            setattr(evaluator, key, value)
        setattr(clf_target, cls.QA_TYPE, True)
        setattr(clf_target, cls.QA_COUNT_TYPE, qa_count)
        session.flush()


    @classmethod
    def enabled_clf_targets(cls):
        """
        The detectors for which we are sending new queries to MTurk for QA
        """
        return ClassifierTarget.query.filter(getattr(ClassifierTarget,
                                                     cls.QA_TYPE) == True).all()


class MTurkVideoDetectorResult(Base, MTurkDetectorResultMixin):

    """Manage results from Mechanical Turk for a given detector"""
    __tablename__ = "mturk_video_detector_results"
    id = Column(Integer, nullable=False, autoincrement=True, primary_key=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False)
    clf_target_id = Column(Integer,
                           ForeignKey("classifier_targets.id"),
                           nullable=False, primary_key=True)
    expected_result = Column(Boolean, nullable=False, default=None)
    hit_id = Column(VARCHAR(length=36), ForeignKey(
        'mturk_video_hits.hit_id'), nullable=False)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())
    clf_target = relation('ClassifierTarget', backref=backref(
        'mturk_video_detector_results', cascade='all,delete-orphan'))
    hit = relation('VideoHit', backref=backref(
        'mturk_video_detector_results', cascade='all,delete-orphan'))
    video = relation('Video', backref=backref(
        'mturk_video_detector_results', cascade='all,delete-orphan'))

    def __unicode__(self):
        return u'classifier_target: %s video_id: %d expected_result: %s' %\
            (self.clf_target.name, self.video_id, self.expected_result)

    # properties for enabling VDR qa for a detector
    QA_TYPE = 'video_qa_enabled'
    QA_COUNT_TYPE = 'collage_count'
    EVALUATOR_CLS = VideoCollageEvaluator

    @classmethod
    def results_to_qa(cls, min_date):
        """
        Find video detector results since (min_date)
        for QA-enabled detectors.
        """
        vdr, wpi = VideoDetectorResult, WebPageInventory
        base_query = session.query(
            vdr.video_id).filter(vdr.timestamp >= min_date)
        base_query = base_query.join((wpi, wpi.video_id == vdr.video_id))
        base_query = base_query.group_by(
            vdr.video_id).order_by(func.sum(wpi.count).desc())

        results = []
        for clf_target in cls.enabled_clf_targets():
            max_hits = clf_target.collage_count
            # Note:
            # Need to use clf.id and not clf_id since
            # clf_id will be int_id for clf whereas
            # for VDRs we use the uuid of the clf.
            query = base_query.filter(vdr.clf_target_id == clf_target.id)
            query = query.outerjoin(
                VideoHit, and_(VideoHit.video_id == vdr.video_id,
                               VideoHit.label_id == clf_target.target_label_id)).\
                filter(VideoHit.hit_id == None)
            results.extend((row[0], clf_target)
                           for row in query.limit(max_hits))
        return results

    @classmethod
    def submit_hits(cls, days=30):
        """Submit video detector results to MTurk for QA"""
        timeframe = timedelta(days)
        min_date = datetime.now() - timeframe
        evaluators = {}
        num_hits_submitted = 0
        for video_id, clf_target in cls.results_to_qa(min_date):
            if clf_target.id not in evaluators:
                evaluators[clf_target.id] = VideoCollageEvaluator.query.\
                    filter_by(target_label_id=clf_target.target_label_id).one()
            evaluator = evaluators[clf_target.id]
            label_id = evaluator.target_label_id
            vh = VideoHit.query.filter_by(video_id=video_id,
                                          label_id=label_id).first()
            if not vh:
                hit_id = evaluator.create_hit(video_id=video_id)
                vh = VideoHit(
                    hit_id=hit_id, label_id=label_id, video_id=video_id)
                num_hits_submitted += 1
                session.flush()

            if not cls.query.filter_by(video_id=video_id,
                                       clf_target_id=clf_target.id).count():
                cls(video_id=video_id, clf_target_id=clf_target.id,
                    hit_id=vh.hit_id, expected_result=True)

        session.flush()
        return VideoHit, num_hits_submitted

    @classmethod
    def _generate_QA_numbers(cls, cls_to_qa_prop, cls_to_qa, join_name, target_result,
                               start_date, end_date):
        count_bools = lambda expr: func.count(func.nullif(expr, 0))
        trues_right = count_bools(
            and_(target_result == True, cls.expected_result == True))
        trues_total = count_bools(
            and_(cls.expected_result == True, target_result != None))
        conflict_total = count_bools(target_result == None)
        query = session.query(
            cls_to_qa_prop, trues_right, trues_total, conflict_total)
        query = query.filter(
            cls.timestamp >= start_date, cls.timestamp < end_date)
        query = query.join(join_name).join(
            cls.hit).filter_by(outstanding=False)
        return query.group_by(cls_to_qa.id).order_by(cls_to_qa.id).all()

    @classmethod
    def generate_QA_numbers(cls, start_date, end_date):
        new_qa_num = []
        qa_num = cls._generate_QA_numbers(
            ClassifierTarget.id, ClassifierTarget,
            ClassifierTarget.mturk_video_detector_results,
            VideoHit.result, start_date, end_date
        )
        for clf_target_id, trues_right, trues_total, conflict_total in qa_num:
            new_qa_num.append((clf_target_id, trues_right, trues_total, conflict_total))
        return new_qa_num


class MTurkBoxDetectorResult(Base, MTurkDetectorResultMixin):

    """ manages results from mechanical Turk for Boxes (QA for Boxes)"""
    __tablename__ = "mturk_box_detector_results"
    box_id = Column(
        Integer, ForeignKey('boxes.id'), nullable=False, primary_key=True)
    clf_target_id = Column(Integer, ForeignKey('classifier_targets.id'),
                           nullable=False, primary_key=True)
    mturk_box_id = Column(
        Integer, ForeignKey('mturk_training_boxes.id'), nullable=False)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    NUM_BOXES_PER_HIT = 21

    clf_target = relation('ClassifierTarget', backref=backref(
        'mturk_box_detector_results', cascade='all,delete-orphan'))
    mturk_box = relation('MTurkBox', backref=backref(
        'mturk_box_detector_results', passive_deletes=True))
    box = relation('Box', backref=backref(
        'mturk_box_detector_results', passive_deletes=True))

    # properties for enabling VDR qa for a detector
    QA_TYPE = 'box_qa_enabled'
    QA_COUNT_TYPE = 'box_qa_count'
    EVALUATOR_CLS = ClickableBoxEvaluator

    def __unicode__(self):
        return u'MTurk QA for Box_id:%s Detector:%s' %\
            (self.box_id, self.detector.name)

    @classmethod
    def results_to_qa(cls, min_date):
        """
        Find (max_hits_per_detector) box_detector_results since min_date
        for Box-QA-enabled Detectors.
        """
        bdr, vdr = BoxDetectorResult, VideoDetectorResult
        results = defaultdict(list)
        for clf_target in cls.enabled_clf_targets():
            max_boxes = clf_target.box_qa_count
            max_videos = max_boxes / BOXES_TO_QA_PER_VIDEO
            if isinstance(clf_target.clf, FaceRecognizeClassifier):
                query = Video.query.join(Video.video_detector_results).filter(
                    vdr.timestamp > min_date,
                    vdr.clf_target_id == clf_target.id)
                query = query.order_by(vdr.timestamp.desc()).limit(max_videos)
                for video in query:
                    box_ids = [
                        b.id for b in video.boxes
                        if b.timestamp in video.s3_timestamps()
                    ]
                    if box_ids:
                        query = session.query(bdr.box_id).filter(
                            bdr.box_id.in_(box_ids),
                            bdr.clf_target_id == clf_target.id)
                        # removing already qa'd boxes
                        query = query.outerjoin(
                            MTurkBox,
                            and_(MTurkBox.box_id == bdr.box_id,
                                 MTurkBox.label_id == clf_target.target_label_id)
                        )
                        query = query.filter(MTurkBox.box_id == None)
                        query = query.order_by(
                            bdr.box_id).limit(BOXES_TO_QA_PER_VIDEO)
                        results[clf_target] += [bid for (bid,) in query]
            else:
                bdr = BoxDetectorResult
                base_query = session.query(Box.id, Box.timestamp, Video)
                base_query = base_query.filter(Box.video_id == Video.id)
                base_query = base_query.join(
                    (bdr, Box.id == bdr.box_id)).filter(
                    bdr.timestamp > min_date,
                    bdr.clf_target_id == clf_target.id)
                # removing already qa'd boxes
                base_query = base_query.outerjoin(
                    MTurkBox,
                    and_(MTurkBox.box_id == Box.id,
                         MTurkBox.label_id == clf_target.target_label_id)
                )
                base_query = base_query.filter(MTurkBox.box_id == None)
                base_query = base_query.order_by(bdr.box_id)
                bids = []
                for bid, ts, v in base_query:
                    if ts in v.s3_timestamps():
                        bids.append(bid)
                    if len(bids) >= max_boxes:
                        break
                results[clf_target] += bids
        return results

    @classmethod
    def submit_hits(cls, days_since_today=30):
        """submit hits to Mturk for all boxes that we have verdict for"""
        min_date = datetime.now() - timedelta(days=days_since_today)
        num_hits_submitted = 0
        results = cls.results_to_qa(min_date)
        for clf_target, box_ids in results.iteritems():
            label_id = clf_target.target_label_id
            evaluator = ClickableBoxEvaluator.query.filter_by(
                target_label_id=label_id).one()
            box_ids_to_submit = []
            for box_id in box_ids:
                mtb = MTurkBox.query.filter_by(label_id=label_id, box_id=box_id).first()
                if mtb:
                    cls(box_id=box_id, clf_target_id=clf_target.id, mturk_box=mtb)
                else:
                    box_ids_to_submit.append(box_id)
            for i in xrange(0, len(box_ids_to_submit), cls.NUM_BOXES_PER_HIT):
                # slicing all boxes so that we can put "NUM_BOXES_PER_HIT"
                # on each BoxHit
                boxes_per_hit = box_ids_to_submit[i:i + cls.NUM_BOXES_PER_HIT]
                hit_id = evaluator.create_hit(box_ids=boxes_per_hit)
                session.flush()
                b = BoxHit(hit_id=hit_id)
                num_hits_submitted += 1
                session.flush()
                for box_id in boxes_per_hit:
                    mtb = MTurkBox(box_id=box_id, box_hit_id=b.id, label_id=label_id)
                    cls(box_id=box_id, clf_target_id=clf_target.id, mturk_box=mtb)
            session.flush()
        return BoxHit, num_hits_submitted

    @classmethod
    def generate_QA_numbers(cls, start_date, end_date):
        """ Generate the QA report for all detectors """
        count_bools = lambda expr: func.count(func.nullif(expr, 0))
        # Total MTurk responses that were True
        trues_right = count_bools(MTurkBox.result == True)
        # Total Responses we got from MTurk (True or False)
        results_total = count_bools(MTurkBox.result != None)
        # Total responses where there was no consensus
        conflict_total = count_bools(MTurkBox.result == None)

        cols = [ClassifierTarget.id,
                trues_right,
                results_total,
                conflict_total]

        query = session.query(*cols)
        query = query.join(ClassifierTarget.target_label)
        query = query.join(ClassifierTarget.mturk_box_detector_results)

        query = query.filter(cls.timestamp >= start_date, cls.timestamp < end_date)
        query = query.join(cls.mturk_box)
        query = query.outerjoin(
            TrainingBox,
            (ClassifierTarget.clf_id == TrainingBox.detector_id) &\
            (MTurkBox.box_id == TrainingBox.box_id))
        query = query.filter(TrainingBox.box_id == None)
        query = query.join(MTurkBox.hit)
        query = query.filter_by(outstanding=False)
        query = query.group_by(cls.clf_target_id).order_by(Label.name)
        return query.all()


class MTurkTextDetectorResult(Base, MTurkDetectorResultMixin):

    """ Recording all the Web Pages we submit for QA  based on Text """
    __tablename__ = "mturk_text_detector_results"
    id = Column(Integer, nullable=False, autoincrement=True, primary_key=True)
    clf_target_id = Column(Integer, ForeignKey('classifier_targets.id'),
                           nullable=False, primary_key=True)
    detector_version = Column(DateTime, nullable=False)
    page_id = Column(Integer, ForeignKey('web_pages.id'), nullable=False)
    hit_id = Column(VARCHAR(128), ForeignKey(
        'mturk_page_hits.hit_id'), nullable=False)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    clf_target = relation('ClassifierTarget', backref=backref(
        'mturk_text_detector_results', cascade='all,delete-orphan'))
    page = relation('WebPage', backref=backref(
        'mturk_text_detector_results', cascade='all,delete-orphan'))
    hit = relation('PageHit', backref=backref(
        'mturk_text_detector_results', cascade='all,delete-orphan'))

    # properties for enabling VDR qa for a detector
    QA_TYPE = 'page_qa_enabled'
    QA_COUNT_TYPE = 'screenshot_count'
    EVALUATOR_CLS = WebPageTextEvaluator

    def __unicode__(self):
        return u'MTurk Text Detector QA for page_id: %d and Detector: %s' % (
            self.page_id, self.detector.name)

    @classmethod
    def results_to_qa(cls, min_date):
        """
        Find (max_hits_per_detector) text detector results since (min_date)
        for QA-enabled detectors.
        """
        tdr, wpi = TextDetectorResult, WebPageInventory
        base_query = session.query(tdr).join(
            (WebPage, tdr.page_id == WebPage.id))
        base_query = base_query.filter(tdr.timestamp >= min_date)
        base_query = base_query.outerjoin((wpi, wpi.page_id == tdr.page_id))

        results = []
        for clf_target in cls.enabled_clf_targets():
            query = base_query.filter(tdr.clf_target_id == clf_target.id,
                                      WebPage.text_detection_update > clf_target.clf.updated_at)
            query = query.outerjoin(
                PageHit, and_(PageHit.page_id == tdr.page_id,
                              PageHit.label_id == clf_target.target_label_id)).\
                filter(PageHit.hit_id == None)
            query = query.order_by(wpi.count.desc())
            query = query.limit(clf_target.screenshot_count)
            for inst in query:
                results.append((clf_target,
                               inst.page_id,
                               clf_target.clf.updated_at))

        return results

    @classmethod
    def register_prev_qa(cls):
        tdr, ph, wp = TextDetectorResult, PageHit, WebPage
        base_query = session.query(
            tdr.page_id, tdr.clf_target_id, ph.hit_id).\
            filter(ph.page_id == tdr.page_id)
        base_query = base_query.join(wp, tdr.page_id == wp.id)
        for clf_target in cls.enabled_clf_targets():
            query = base_query.\
                filter(tdr.clf_target_id == clf_target.id,
                       wp.text_detection_update > clf_target.clf.updated_at,
                       ph.label_id == clf_target.target_label_id)
            query = query.\
                outerjoin(cls, and_(cls.clf_target_id == clf_target.id,
                                    cls.detector_version == clf_target.clf.updated_at,
                                    cls.page_id == tdr.page_id))
            for i in query.filter(cls.page_id == None):
                cls(page_id=i.page_id, detector_version=clf_target.clf.updated_at,
                    clf_target_id=clf_target.id, hit_id=i.hit_id)
        session.flush()

    @classmethod
    def submit_hits(cls, days=30):
        """Submit web pages to MTurk for QA"""
        min_date = datetime.now() - timedelta(days)
        evaluators = {}
        cls.register_prev_qa()
        num_hits_submitted = 0
        for clf_target, page_id, clf_updated_at in cls.results_to_qa(min_date):
            if clf_target.id not in evaluators:
                label_id = clf_target.target_label_id
                evaluators[clf_target.id] = WebPageTextEvaluator.query.filter_by(
                    target_label_id=label_id).one()
            evaluator = evaluators[clf_target.id]
            label_id = evaluator.target_label_id
            ph = PageHit.query.filter_by(
                page_id=page_id, label_id=label_id).first()
            if not ph:
                hit_id = evaluator.create_hit(page_id=page_id)
                ph = PageHit(hit_id=hit_id, label_id=label_id, page_id=page_id)
                num_hits_submitted += 1
            cls(page_id=page_id, detector_version=clf_updated_at,
                clf_target_id=clf_target.id, hit_id=ph.hit_id)
            session.flush()
        return PageHit, num_hits_submitted

    @classmethod
    def _get_training_data(cls):
        """ Returns page-ids used for training detectors for the latest version in (detector_id, page_id) format """
        tp = TrainingPage
        # Get latest versions for all detector-ids
        detector_id_versions = []
        for (dtc_id,) in session.query(tp.detector_id).distinct(tp.detector_id):
            latest_version = AbstractTextDetector.get(dtc_id).updated_at
            detector_id_versions.append((dtc_id, latest_version))

        if not detector_id_versions:
            return []

        # Get training page-ids only for the latest versions
        tr_dtc_page_ids = session.query(tp.detector_id, tp.page_id).filter(
            tuple_(tp.detector_id, tp.detector_version).in_(detector_id_versions))
        return tr_dtc_page_ids.all()

    @classmethod
    def generate_QA_numbers(cls, start_date, end_date):
        tr_dtc_page_ids = cls._get_training_data()
        target_result = PageHit.result
        count_bools = lambda expr: func.count(func.nullif(expr, 0))
        trues_right = count_bools(target_result == True)
        trues_total = count_bools(target_result != None)
        conflict_total = count_bools(target_result == None)
        query = session.query(
            ClassifierTarget.id, trues_right, trues_total, conflict_total)
        query = query.join(AbstractTextDetector,
                           ClassifierTarget.clf_id==AbstractTextDetector.id)
        query = query.filter(
            cls.timestamp >= start_date, cls.timestamp < end_date,
            cls.detector_version == AbstractTextDetector.updated_at)
        query = query.join(ClassifierTarget.mturk_text_detector_results)\
                     .join(cls.hit).filter_by(outstanding=False)
        if tr_dtc_page_ids:
            query = query.filter(
                ~tuple_(AbstractTextDetector.id, cls.page_id).in_(tr_dtc_page_ids))
        # results = query.group_by(ClassifierTarget.id).order_by(ClassifierTarget.id).all()
        # return [tuple([r[0].name] + list(r[1:])) for r in results]
        query = query.group_by(ClassifierTarget.id).order_by(ClassifierTarget.id)
        return query.all()


class MTurkImageDetectorResult(Base, MTurkDetectorResultMixin):

    """ manages results from mechanical Turk for Images (QA for Images)"""
    NUM_IMAGES_PER_HIT = 21

    __tablename__ = "mturk_image_detector_results"
    id = Column(Integer, nullable=False, autoincrement=True, primary_key=True)
    mturk_image_id = Column(
        Integer, ForeignKey('mturk_images.id'), nullable=False)
    clf_target_id = Column(Integer, ForeignKey('classifier_targets.id'),
                           nullable=False, primary_key=True)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    clf_target = relation('ClassifierTarget', backref=backref(
        'mturk_image_detector_results', cascade='all,delete-orphan'))
    mturk_image = relation('MTurkImage', backref=backref(
        'mturk_image_detector_results', cascade='all,delete-orphan'))

    # properties for enabling VDR qa for a detector
    QA_TYPE = 'image_qa_enabled'
    QA_COUNT_TYPE = 'image_qa_count'
    EVALUATOR_CLS = ClickableImageEvaluator

    def __unicode__(self):
        return u'MTurk QA for Image_id:%s Clf-Target:%s' % (self.mturk_image_id,
                                                            self.clf_target.name)
    @classmethod
    def results_to_qa(cls, min_date):
        """ Find image_detector_results since min_date for
        Image-QA-enabled Clf Targets"""
        idr, ti = ImageDetectorResult, TrainingImage
        results = defaultdict(set)

        for clf_target in cls.enabled_clf_targets():
            # Image count ends up being the number of videos from which we
            # query. So, the actual number of images might be more
            max_videos = clf_target.image_qa_count
            query = session.query(idr.video_id.distinct())
            query = query.filter(idr.clf_target_id == clf_target.id,
                                 idr.timestamp > min_date)
            myvids = set()

            for (vid,) in query:
                if(len(myvids) >= max_videos):
                    break
                s3_images = Video.get(vid).s3_timestamps()
                if s3_images:
                    query = session.query(idr.video_id, idr.time).filter(
                        idr.video_id == vid, idr.clf_target_id == clf_target.id,
                        idr.time.in_(s3_images))
                    # Filter images used for training the detector
                    training_images = session.query(
                        ti.video_id, ti.timestamp).\
                        filter_by(detector_id=clf_target.clf.id).all()
                    for vid_id, ts in query:
                        if (vid_id, ts) not in training_images:
                            mtb = MTurkImage.query.filter_by(
                                video_id=vid_id, timestamp=ts,
                                label_id=clf_target.target_label_id).first()
                            if not mtb:
                                results[clf_target].add((vid_id, ts))
                                myvids.add(vid_id)

        return results

    @classmethod
    def submit_hits(cls, days_since_today=30):
        """submit hits to Mturk for all images that we have verdict for"""

        min_date = datetime.utcnow() - timedelta(days=days_since_today)
        num_hits_submitted = 0
        for clf_target, images in cls.results_to_qa(min_date).iteritems():
            label_id = clf_target.target_label_id
            evaluator = ClickableImageEvaluator.query.filter_by(
                target_label_id=label_id).one()
            images_to_submit = []

            for vid, ts in images:
                mtb = MTurkImage.query.filter_by(
                    video_id=vid, timestamp=ts, label_id=label_id).first()
                if not mtb:
                    # post HIT if no MTurkImage
                    images_to_submit.append([vid, ts])

            for i in xrange(0, len(images_to_submit), cls.NUM_IMAGES_PER_HIT):
                images_per_hit = images_to_submit[i:i + cls.NUM_IMAGES_PER_HIT]
                hit_id = evaluator.create_hit(image_ids=images_per_hit)
                ih = ImageHit(hit_id=hit_id)
                num_hits_submitted += 1
                session.flush()

                for vid, ts in images_per_hit:
                    mti = MTurkImage(
                        video_id=vid, timestamp=ts, image_hit_id=ih.id,
                        label_id=label_id)
                    cls(clf_target_id=clf_target.id, mturk_image=mti)
                session.flush()
        return ImageHit, num_hits_submitted

    @classmethod
    def generate_QA_numbers(cls, start_date, end_date):
        """ generate the QA report for specified clf-target """
        count_bools = lambda expr: func.count(func.nullif(expr, 0))

        # Total MTurk responses that were True
        trues_right = count_bools(MTurkImage.result == True)
        # Total Responses we got from MTurk (True or False)
        results_total = count_bools(MTurkImage.result != None)
        # Total responses where there was no consensus
        conflict_total = count_bools(MTurkImage.result == None)

        query = session.query(
            ClassifierTarget.id, trues_right, results_total, conflict_total)
        query = query.filter(
            cls.timestamp >= start_date, cls.timestamp < end_date)
        query = query.join(ClassifierTarget.mturk_image_detector_results)
        query = query.join(MTurkImage, MTurkImage.id == cls.mturk_image_id)
        query = query.join(
            ImageHit, ImageHit.id == MTurkImage.image_hit_id).\
            filter_by(outstanding=False)
        query = query.group_by(cls.clf_target_id).order_by(ClassifierTarget.id)
        # return [tuple([r[0].name] + list(r[1:])) for r in query]
        return query.all()


class MTurkVideoResult(Base):
    __tablename__ = "mturk_video_results"

    video_id = Column(
        Integer, ForeignKey('videos.id'), nullable=False, primary_key=True)
    evaluator_id = Column(Integer, ForeignKey(
        'mturk_evaluators.id'), nullable=False, primary_key=True)
    result = Column(Boolean, nullable=False)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    evaluator = relation('MechanicalTurkEvaluator',
                         backref=backref('video_results', passive_deletes=True))
    video = relation('Video', backref=backref(
        'mturk_results', passive_deletes=True))

    def __unicode__(self):
        return u'<%s mturk video evaluation (%s) (%s)>' %\
            (self.result, self.evaluator.name, self.video.id)

    @classmethod
    def set_result(cls, video_id, evaluator_id, result):
        cols = 'video_id, evaluator_id, result'
        execute("""
            insert into %s (%s)
                values (%s, "%s", %s)
                on duplicate key update result = %s""" %
                (cls.__tablename__, cols, video_id, evaluator_id, int(result),
                 int(result)))

