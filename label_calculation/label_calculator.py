"""Logic for determining label/ keyword results to record for a given page.
Also, implementations of bulk calculation and recording of labels
for groups of pages.
"""
from collections import defaultdict
import csv
from datetime import datetime
import logging
from tempfile import NamedTemporaryFile
import time

from sqlalchemy.orm import joinedload

from affine import config
from affine.aws import sqs
from affine.model.base import session
from affine.model.admin_decisions import AdminVideoLabelResult, \
    AdminWebPageLabelResult
from affine.model.detection import AbstractClassifier, \
    TextDetectorResult, DomainNameDetector, VideoDetectorResult
from affine.model.labels import Label, Keyword, WeightedKeyword, \
    WeightedClfTarget, WeightedLabel
from affine.model.mturk.results import MTurkVideoResult
from affine.model.mturk.evaluators import MechanicalTurkEvaluator
from affine.model.url_blacklist import RotatingContentPage, UserInitiatedDomain
from affine.model.videos import Video
from affine.model.web_page_label_results import WebPageLabelResult
from affine.model.web_pages import WebPage, get_page_processed_text_dict, VideoOnPage
from affine.model.classifier_target_labels import ClassifierTarget
from affine.normalize_url import domain_of_url
from affine.retries import retry_operation
from affine.vcr.dynamodb import DynamoIngestionStatusClient

__all__ = ['save_labels_for_pages']

logger = logging.getLogger(__name__)

ALL_UKS = 'ALL_UKS'


class LabelCalculator(object):

    """Bulk calculator of label/ keyword results for a set of web pages"""

    def __init__(self, page_ids, label_ids=None, start_time=None):
        self.page_ids = list(page_ids)
        self.all_labels = label_ids is None
        self.target_label_ids = set(label_ids or [])
        self.start_time = start_time or datetime.utcnow()

        if self.all_labels:
            logger.info('Recalculating for all labels')
        elif self.target_label_ids:
            label_str = ','.join(map(str, label_ids))
            logger.info('Recalculating for %s labels : %s' %
                        (len(label_ids), label_str[:10] + '...'))
        else:
            raise ValueError("No Labels to Recalc")

    @property
    def log_description(self):
        log_desc = ','.join(map(str, self.page_ids[:5]))
        if len(self.page_ids) > 5:
            log_desc += '...(%d)' % len(self.page_ids)        
        return log_desc
    
    def prefetch_from_db(self):
        """For performance reasons,
        we don't want to do database queries for every page
        Pull relevant information from the DB that we will need later and
        store it as dicts and sets.
        """
        logger.info('querying db for %s', self.log_description)
        self.label_ids_by_name = {}
        self.text_detector_weights = {}
        self.label_lookup = {}

        self.prefetch_keywords()
        self.prefetch_generic_labels()
        self.prefetch_labels()
        
        self.remote_ids = dict(session.query(
            WebPage.id, WebPage.remote_id).filter(WebPage.id.in_(self.page_ids)))
        self.rotating_content_pages = [
            row.remote_id for row in session.query(RotatingContentPage.remote_id)]
        
        self.prefetch_active_videos()
        self.prefetch_crawl_status()
        self.prefetch_last_detection_times()
        self.prefetch_page_contents()
        self.prefetch_wt_clf_targets()
        self.prefetch_vdrs()
        self.prefetch_wplrs()
        self.prefetch_tdrs()
        self.prefetch_domain_results()
        self.prefetch_admin_labels()
        logger.info('done querying db for %s', self.log_description)

    def prefetch_generic_labels(self):
        """Populate self.generic_label_ids"""
        length_label_names = [
            "Short (0-4 min.)",
            "Medium (4-20 min.)",
            "Long (20+ min.)",
        ]
        preroll_labels = [
            "pre-roll",
            "non-pre-roll",
        ]
        other_label_names = [
            "Rotating Content",
        ]

        length_label_ids = {self.label_id_by_name(name) for name in length_label_names}
        other_label_ids = {self.label_id_by_name(name) for name in other_label_names}
        preroll_label_ids = {self.label_id_by_name(name) for name in preroll_labels}
        self.generic_label_ids = length_label_ids | other_label_ids | preroll_label_ids
        if not self.all_labels:
            self.generic_label_ids = self.generic_label_ids & self.target_label_ids
            length_label_count = len(self.target_label_ids & length_label_ids)
            assert length_label_count in {0, 3}, 'Must have all or no length labels'
            preroll_target_label_ids = self.target_label_ids & preroll_label_ids
            assert not preroll_target_label_ids or\
                preroll_target_label_ids == preroll_label_ids,\
                'Must have all or no pre-roll labels'

    def prefetch_labels(self):
        """Populate self.base_label_ids"""
        self.base_label_ids = set()
        self.label_decision_thresholds = {}

        if not self.all_labels:    
            # use only labels that can actually produce results
            # i.e. have at least one weighted_keyword, weighted_label,
            # weighted_detector or weighted_text_detector
            query = session.query(Label.id.distinct()).filter(
                Label.id.in_(self.target_label_ids))
            query = query.outerjoin(Label.weighted_keywords)
            query = query.outerjoin(Label.weighted_labels)
            query = query.outerjoin(Label.weighted_clf_targets)
            query = query.filter(
                (WeightedKeyword.keyword_id != None) |
                (WeightedLabel.child_id != None) |
                (WeightedClfTarget.clf_target_id != None))
            self.base_label_ids.update(row[0] for row in query)
        else:
            query = session.query(Label.id.distinct())
            query = query.outerjoin(Label.weighted_keywords)
            query = query.outerjoin(Label.weighted_clf_targets)
            query = query.filter(
                (WeightedKeyword.keyword_id != None) |
                (WeightedClfTarget.clf_target_id != None))
            label_ids = [row[0] for row in query]
            if label_ids:
                self.base_label_ids.update(Label.all_ancestor_ids(label_ids))

        self.descendant_label_ids = (Label.all_descendant_ids(self.target_label_ids)
                                     - self.target_label_ids)
        
        # fetch all label thresholds, since it is cheap
        query = session.query(Label.id, Label.decision_threshold)
        for label_id, thresh in query:
            self.label_decision_thresholds[label_id] = thresh

    def prefetch_keywords(self):
        """Populate self.keywords"""
        from affine.detection.nlp.keywords.keyword_matching import \
            PageKeywordMatcher
        
        self.keyword_matcher = PageKeywordMatcher()
        self.weighted_keywords = defaultdict(list)

        query = session.query(WeightedKeyword)
        query = query.options(joinedload(WeightedKeyword.keyword))
        if not self.all_labels:
            query = query.filter(WeightedKeyword.label_id.in_(self.target_label_ids))
        for wk in query:
            kw = wk.keyword
            self.weighted_keywords[wk.label_id].append((kw.id, wk.title_weight, wk.body_weight))
            self.keyword_matcher.add_keyword(kw.id, kw.text)

    def prefetch_page_contents(self):
        """Populate self.page_contents"""
        self.page_contents = {}
        query = session.query(WebPage.id, WebPage.processed_title).filter(WebPage.id.in_(self.page_ids))
        processed_text_dict = get_page_processed_text_dict(self.page_ids, silent=True)
        for page_id, processed_title in query:
            self.page_contents[page_id] = (processed_title, processed_text_dict[page_id])

    def prefetch_wt_clf_targets(self):
        self.wt_clf_target_lookup = defaultdict(list)
        self.clf_target_ids = set()

        cols = [WeightedClfTarget.label_id,
                WeightedClfTarget.clf_target_id,
                WeightedClfTarget.weight]
        query = session.query(*cols)
        if not self.all_labels:
            query = query.filter(
                WeightedClfTarget.label_id.in_(self.target_label_ids))
        query = query.join(WeightedClfTarget.clf_target)
        query = query.join(ClassifierTarget.clf)
        query = query.filter(AbstractClassifier.enabled_since != None)

        for label_id, clf_target_id, weight in query:
            self.wt_clf_target_lookup[label_id].append(
                (clf_target_id, weight))
            self.clf_target_ids.add(clf_target_id)

    def prefetch_vdrs(self):
        # lookup containing true detectors for videos
        self.vdr_lookup = defaultdict(set)

        # gather all active videos
        video_ids = set()
        for video_list in self.active_videos.values():
            for vcr_video in video_list:
                video_ids.add(vcr_video['video_id'])

        # get all video detector results
        if self.clf_target_ids and video_ids:
            vdr = VideoDetectorResult
            query = session.query(vdr.video_id, vdr.clf_target_id)
            query = query.filter(vdr.video_id.in_(video_ids))
            query = query.filter(vdr.clf_target_id.in_(self.clf_target_ids))
            for video_id, clf_target_id in query:
                self.vdr_lookup[video_id].add(clf_target_id)

    def prefetch_wplrs(self):
        self.wplr_lookup = defaultdict(set)
        query = session.query(
            WebPageLabelResult.page_id, WebPageLabelResult.label_id)
        query = query.filter(WebPageLabelResult.page_id.in_(self.page_ids))

        if not self.all_labels:
            label_ids = Label.all_descendant_ids(self.target_label_ids)
            query = query.filter(WebPageLabelResult.label_id.in_(label_ids))

        for page_id, label_id in query:
            self.wplr_lookup[page_id].add(label_id)

    def prefetch_avlrs(self):
        """Grab all admin video label results for our pages"""
        self.avlr_lookup = defaultdict(list)

        query = session.query(VideoOnPage.page_id,
                              AdminVideoLabelResult.label_id,
                              AdminVideoLabelResult.result)
        query = query.join((AdminVideoLabelResult,
                           VideoOnPage.video_id == AdminVideoLabelResult.video_id))
        query = query.filter(VideoOnPage.active == True,
                             VideoOnPage.is_preroll == False,
                             VideoOnPage.page_id.in_(self.page_ids))
        if not self.all_labels:
            query = query.filter(
                AdminVideoLabelResult.label_id.in_(self.target_label_ids))

        for page_id, label_id, result in query:
            self.avlr_lookup[page_id].append((label_id, result))

    def prefetch_awplrs(self):
        """Grab all admin web page label results for our pages"""
        self.awplr_lookup = defaultdict(list)

        query = session.query(AdminWebPageLabelResult.page_id,
                              AdminWebPageLabelResult.label_id,
                              AdminWebPageLabelResult.result)
        query = query.filter(
            AdminWebPageLabelResult.page_id.in_(self.page_ids))
        if not self.all_labels:
            query = query.filter(
                AdminWebPageLabelResult.label_id.in_(self.target_label_ids))

        for page_id, label_id, result in query:
            self.awplr_lookup[page_id].append((label_id, result))

    def prefetch_svdrs(self):
        """Grab all MTurk super results for our pages"""
        self.svdr_lookup = defaultdict(list)

        query = session.query(VideoOnPage.page_id,
                              MechanicalTurkEvaluator.target_label_id,
                              MTurkVideoResult.result)
        query = query.join(
            (MTurkVideoResult, VideoOnPage.video_id == MTurkVideoResult.video_id))
        query = query.join((MechanicalTurkEvaluator,
                            MechanicalTurkEvaluator.id == MTurkVideoResult.evaluator_id))
        query = query.filter(MechanicalTurkEvaluator.super == True)
        query = query.filter(VideoOnPage.active == True,
                             VideoOnPage.is_preroll == False,
                             VideoOnPage.page_id.in_(self.page_ids))
        if not self.all_labels:
            query = query.filter(
                MechanicalTurkEvaluator.target_label_id.in_(self.target_label_ids))

        for page_id, label_id, result in query:
            self.svdr_lookup[page_id].append((label_id, result))

    def active_video_for_page(self, page_id):
        videos = self.active_videos[page_id]
        if videos:
            return videos[0]

    def prefetch_active_videos(self):
        """Grab all VideoOnPage objects for active videos on our pages"""
        self.active_videos = defaultdict(list)
        
        cols = [VideoOnPage.page_id, VideoOnPage.video_id, 
                Video.length, Video.last_detection]
        
        query = session.query(*cols)
        query = query.filter(
            VideoOnPage.active == True, 
            VideoOnPage.is_preroll == False, 
            VideoOnPage.page_id.in_(self.page_ids))
        query = query.join(VideoOnPage.video)
        query = query.order_by(VideoOnPage.id.desc())

        for page_id, video_id, length, last_detection in query:
            vcr_video = {
                "video_id": video_id,
                "length": length,
                "last_detection" : last_detection
            }
            self.active_videos[page_id].append(vcr_video)

    def prefetch_last_detection_times(self):
        self.last_detections = defaultdict(lambda: None)
        self.last_text_detections = defaultdict(lambda: None)

        for page_id in self.page_ids:
            vcr_videos = self.active_videos[page_id]

            last_detections = [vv['last_detection'] for vv in vcr_videos]
            # Set it to min of last_detections or None
            if last_detections == [] or None in last_detections:
                last_detection = None
            else:
                last_detection = min(last_detections)
            self.last_detections[page_id] = last_detection

        query = session.query(WebPage.id, WebPage.text_detection_update)
        query = query.filter(WebPage.id.in_(self.page_ids))
        self.last_text_detections.update(query)

    def prefetch_crawl_status(self):
        self.video_crawl_complete = defaultdict(bool)

        q = session.query(WebPage.id, WebPage.last_crawled_video)
        q = q.filter(WebPage.id.in_(self.page_ids))

        for page_id, video_ts in q:
            if video_ts is not None:
                self.video_crawl_complete[page_id] = True

    def prefetch_tdrs(self):
        """Grab all Text Detector Results for our pages"""
        # lookup containing text detector results for pages
        self.tdr_lookup = defaultdict(set)

        if self.clf_target_ids:
            tdr = TextDetectorResult
            query = session.query(tdr.page_id, tdr.clf_target_id)
            query = query.filter(tdr.page_id.in_(self.page_ids))
            query = query.filter(tdr.clf_target_id.in_(self.clf_target_ids))

            for page_id, clf_target_id in query:
                self.tdr_lookup[page_id].add(clf_target_id)

    def prefetch_domain_results(self):
        self.domain_lookup = defaultdict(int)

        dnd = DomainNameDetector
        query = session.query(dnd.domain_name, dnd.target_label_id, dnd.weight)

        if not self.all_labels:
            query = query.filter(
                dnd.target_label_id.in_(self.target_label_ids))

        domain_name_detectors = query.all()

        for page_id in self.page_ids:
            page_url = self.remote_ids[page_id]
            page_domain = domain_of_url(page_url, with_subdomains=True)
            for domain_name, label_id, weight in domain_name_detectors:
                # Both page_domain and domain_name are prefixed with a dot
                # so that this does the right thing for subdomains.
                # For example, page domain '.a.b.com' will match a dnd for
                # '.b.com' or '.a.b.com', but not '.sports.a.b.com'
                # or '.sportsa.b.com'
                if page_domain.endswith(domain_name):
                    self.domain_lookup[(page_id, label_id)] += weight
    
    def prefetch_admin_labels(self):
        # todo : combine all admins into single dict from get go
        # instead of creating 3 temp data lookups
        self.prefetch_avlrs()
        self.prefetch_awplrs()
        self.prefetch_svdrs()
        
    def calculate(self, page_id):
        """Determine the keyword and label matches
        for a single page from our pages"""
        labels_to_add = set()
        labels_to_delete = set()
        labels_to_add, labels_to_delete = self.calculate_labels(page_id)
        return labels_to_add, labels_to_delete

    def calculate_info_labels(self, page_id):
        """Calculate all info based labels that are not part of the hierarchy"""
        rotating_content_label_id = self.label_id_by_name("Rotating Content")
        short_label_id = self.label_id_by_name("Short (0-4 min.)")
        medium_label_id = self.label_id_by_name("Medium (4-20 min.)")
        long_label_id = self.label_id_by_name("Long (20+ min.)")
        preroll_label_id = self.label_id_by_name("pre-roll")
        non_preroll_label_id = self.label_id_by_name("non-pre-roll")

        vcr_video = self.active_video_for_page(page_id)

        info_labels = set()
        # preroll labels can be applied only after video ingestion has been finished
        # until that we can never be sure if the page had a video or not
        if self.video_crawl_complete[page_id]:
            # if no active video, we add the non-pre-roll label
            if vcr_video is None:
                info_labels.add(non_preroll_label_id)
            else:
                info_labels.add(preroll_label_id)

        if vcr_video is not None:
            remote_id = self.remote_ids[page_id]
            if remote_id in self.rotating_content_pages:
                info_labels.add(rotating_content_label_id)

            minutes = vcr_video['length'] / 60.
            if minutes < 4:
                info_labels.add(short_label_id)
            elif 4 <= minutes < 20:
                info_labels.add(medium_label_id)
            else:
                info_labels.add(long_label_id)

        # return only those labels that were asked for in the first place
        info_labels = info_labels & self.generic_label_ids

        return info_labels

    def calculate_admin_labels(self, page_id):
        # Admin video label results
        avlrs = self.avlr_lookup[page_id]
        # Admin web page label results
        awplrs = self.awplr_lookup[page_id]
        # Super-detector video results
        svdrs = self.svdr_lookup[page_id]

        admin_results = dict(svdrs)
        admin_results.update(avlrs)
        admin_results.update(awplrs)

        false_admin_labels = {l for l, r in admin_results.items() if not r}
        for descendant_id in Label.all_descendant_ids(false_admin_labels):
            admin_results[descendant_id] = False

        return admin_results

    def calculate_labels(self, page_id):
        """Determine the current correct label results for a page
        based on its active videos and keywords on the page. Results are
        returned as set of new label IDs to add and old ones to remove.
        """
        admin_results = self.calculate_admin_labels(page_id)
        
        # gathering all label_ids that we will calculate scores 
        # and pass to calculate_sublabels
        label_ids = self.base_label_ids.copy()
        if self.all_labels:
            admin_label_ids = set(admin_results.keys())
            label_ids.update(Label.all_ancestor_ids(admin_label_ids))

        kw_title_matches, kw_body_matches = self.calculate_keywords(page_id)
        # Calculate label scores based on matching keywords, true clf-targets and domains
        label_scores = dict()

        for label_id in label_ids:
            if label_id not in admin_results:
                score = 0
                for kw_id, title_weight, body_weight in self.weighted_keywords[label_id]:
                    if kw_id in kw_title_matches and title_weight != 0:
                        score += title_weight
                    elif kw_id in kw_body_matches:
                        score += body_weight

                # adding weights for all weighted clf-targets
                for clf_target_id, weight in self.wt_clf_target_lookup[label_id]:
                    if clf_target_id in self.tdr_lookup[page_id]:
                        score += weight
                    # VDRs are considered for all non preroll videos of the page
                    for video in self.active_videos[page_id]:
                        video_id = video['video_id']
                        if clf_target_id in self.vdr_lookup[video_id]:
                            score += weight

                # adding weights from domains
                score += self.domain_lookup[(page_id, label_id)]

                label_scores[label_id] = score

        # Existing Web Page Label Results
        existing_true_label_ids = self.wplr_lookup[page_id]
        
        # existing calculated_true_label_ids are just descendant existing_true_label_ids
        true_label_ids = existing_true_label_ids & self.descendant_label_ids

        calculated_true_label_ids = self.calculate_sublabels(
            label_ids, label_scores, true_label_ids,
            admin_results, self.label_decision_thresholds)
        
        # append all the info labels to True Labels 
        info_labels = self.calculate_info_labels(page_id)
        calculated_true_label_ids |= info_labels

        # here we need to remove existing_true_label_ids that are already in the DB
        # as they do not need to be re-inserted into mysql
        labels_to_add = calculated_true_label_ids - existing_true_label_ids

        if self.all_labels:
            labels_to_delete = existing_true_label_ids - calculated_true_label_ids
        else:
            labels_expected_to_have_results = label_ids | self.target_label_ids
            labels_to_delete = (
                labels_expected_to_have_results - calculated_true_label_ids) & existing_true_label_ids

        return labels_to_add, labels_to_delete

    def calculate_keywords(self, page_id):
        """returns a list of kw ids that are found on the given page"""
        page_title, page_text = self.page_contents[page_id]
        kw_title_matches = self.keyword_matcher.matching_keywords(page_title)
        kw_body_matches = self.keyword_matcher.matching_keywords(page_text)
        return (kw_title_matches, kw_body_matches)

    def calculate_sublabels(self, label_ids, label_scores, true_labels,
                            admin_label_results, label_decision_thresholds):
        """Update true_labels to contain the results for all the given labels

        based on scores it may have for keywords/ detectors in label_scores and
        based on considering its children's results in label_results.
        """
        missing_label_ids = set(label_ids) - set(self.label_lookup.keys())
        if missing_label_ids:
            query = Label.query.filter(Label.id.in_(missing_label_ids))
            query = query.options(joinedload(Label.weighted_labels))
            for label in query:
                weighted_labels = [(wl.child_id, wl.weight)
                                   for wl in label.weighted_labels]
                self.label_lookup[label.id] = {
                    'rank': label.rank, 'weighted_labels': weighted_labels}

        true_labels = set(true_labels)
        for label_id, result in admin_label_results.iteritems():
            if result:
                true_labels.add(label_id)
            else:
                true_labels.discard(label_id)

        label_ids = sorted(label_ids,
                           key=(lambda label_id: self.label_lookup[label_id]['rank']),
                           reverse=True)

        for label_id in label_ids:
            if label_id not in admin_label_results:
                # the scores from the detectors, now looking for results from
                # labels
                score = label_scores.get(label_id, 0)
                for sub_label_id, weight in self.label_lookup[label_id]['weighted_labels']:
                    if sub_label_id in true_labels:
                        score += weight
                if score >= label_decision_thresholds[label_id]:
                    true_labels.add(label_id)

        return true_labels

    def save_results(self):
        """Record results for all of our pages to the DB"""
        pages_with_updates = set()
        wplr_file = NamedTemporaryFile('wb')
        wplr_csv = csv.writer(wplr_file, delimiter="\t")
        
        for page_id in self.page_ids:
            labels_to_add, labels_to_delete = self.calculate(page_id)

            # if there are any changes to the labels on this page
            # add it to updated_pages_queue
            if labels_to_add or labels_to_delete:
                pages_with_updates.add(page_id)

            for label_id in labels_to_add:
                wplr_csv.writerow([page_id, label_id])
            if labels_to_delete:
                logger.info('For page_id: %s, deleting label results for label_ids : %s',
                            page_id,
                            list(labels_to_delete))
                query = WebPageLabelResult.query.filter_by(page_id=page_id)
                query = query.filter(
                    WebPageLabelResult.label_id.in_(labels_to_delete))
                retry_operation(query.delete,
                                synchronize_session=False,
                                error_message='Deleting WPLRs failed')

        wplr_file.flush()
        
        WebPageLabelResult.load_from_file(wplr_file.name)
        with session.begin():
            # update last_label_update for all the pages in the chunk
            query = WebPage.query.filter(WebPage.id.in_(self.page_ids))
            query.update(
                {'last_label_update': self.start_time}, synchronize_session=False)
            for page_id in self.page_ids:
                last_detection = self.last_detections[page_id]
                last_text_detection = self.last_text_detections[page_id]
                query = WebPage.query.filter_by(id=page_id)
                query.update({
                    'last_detection_at_llu': last_detection,
                    'last_text_detection_at_llu': last_text_detection,
                })

        return pages_with_updates

    def update_dynamo(self):
        urls = self.remote_ids.values()
        dynamo = DynamoIngestionStatusClient()
        dynamo_data = dynamo.batch_get(urls)
        to_put = []
        for url in urls:
            item = dynamo_data.get(url)
            if item is None or (item['status'] not in ['Complete', 'Failed']):
                if item is None:
                    # Arbitary old date so we don't treat this as newly ingested
                    # The fact that there is no record in Dynamo means
                    # it wasn't ingested recently
                    # This should never happen because
                    # the only pages that make it here should
                    # already be in Dynamo, but if it does,
                    # try to avoid corrupting our metrics
                    # for recently "ingested" pages
                    created_at = datetime(2011, 1, 1)
                else:
                    created_at = item['created_at']
                to_put.append({
                    'url': url,
                    'status': 'Complete',
                    'created_at': created_at,
                })
        dynamo.batch_put(to_put)

    def label_id_by_name(self, name):
        if name not in self.label_ids_by_name:
            label_id = Label.by_name(name).id
            self.label_ids_by_name[name] = label_id
        return self.label_ids_by_name[name]


def forward_updated_pages(pages_to_forward):
    logger.info('Forwarding %s pages' %len(pages_to_forward))
    data = {
        'page_ids' : list(pages_to_forward),
        'timestamp' : int(time.time()),
    }
    queue_name = config.get('sqs.page_label_updates_queue')
    queue = sqs.get_queue(queue_name)
    sqs.write_to_queue(queue, data)


def save_labels_for_pages(page_ids, label_ids=None, start_time=None):
    """Record label results for a set of pages to the DB"""
    if page_ids:
        calculator = LabelCalculator(
            page_ids, label_ids=label_ids, start_time=start_time)
        calculator.prefetch_from_db()
        pages_with_updates = calculator.save_results()
        calculator.update_dynamo()
        # if there are paegs with labels updates, 
        # forward them to the queue "page_label_updates_queue"
        if pages_with_updates:
            forward_updated_pages(pages_with_updates)
