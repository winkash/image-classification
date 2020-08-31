from logging import getLogger
from datetime import datetime, timedelta

from affine.model.base import *
from affine.model._sqla_imports import *

from affine.model.mturk.jobs import AbstractMTurkLabelJob
from affine.model.mturk.hits import VideoHit, PageHit
from affine.model.labels import Label
from affine.model.mturk.jobs import AbstractMTurkLabelJob
from affine.model.mturk.hits import VideoHit, PageHit
from affine.model.web_pages import WebPage, WebPageInventory
from affine.model.web_page_label_results import WebPageLabelResult
from affine.model.mturk.evaluators import VideoCollageEvaluator, WebPageTextEvaluator
from affine.model.training_data import TrainingPage, LabelTrainingPage

logger = getLogger(__name__)
__all__ = ['MTurkLabelScreenshotJob', 'MTurkLabelCollageJob']

class MTurkLabelScreenshotJob(AbstractMTurkLabelJob):
    """ Job class for submitting screenshot HITs for Labels
    """
    __mapper_args__ = dict(polymorphic_identity='mturk_label_screenshot_jobs')
    preroll_ratio = Column(Float, nullable=True)
    result_table = PageHit

    def get_ignore_page_ids(self):
        """ Returns page ids that were part of training or have already
        been QA'd for the given label or an equivalent one """
        prev_qad_ids = {ph.page_id for ph in PageHit.query.
                        filter(PageHit.label_id==self.label_id)}
        training_ids = {ltp.page_id for ltp in LabelTrainingPage.query.
                        filter_by(label_id=self.label_id)}
        return prev_qad_ids | training_ids

    def _results_to_qa_query(self, page_ids_to_ignore):
        """ Returns a general query object for getting page-ids to be QA'ed which
            are in the inventory and have a WebPageLabelResult for the input label-id

            Returns:
                SQLAlchemy query object
        """
        wpi = WebPageInventory
        wplr = WebPageLabelResult
        query = session.query(wpi.page_id.distinct())
        query = query.join(WebPage, wpi.page_id == WebPage.id)
        query = query.join(wplr, wplr.page_id == wpi.page_id)
        if page_ids_to_ignore:
           query = query.filter(~wpi.page_id.in_(page_ids_to_ignore))
        query = query.filter(wplr.label_id == self.label_id)
        query = query.order_by(func.rand())
        return query

    def results_to_qa_all(self, limit):
        page_ids_to_ignore = self.get_ignore_page_ids()
        query = self._results_to_qa_query(page_ids_to_ignore).limit(limit)
        return [page_id for (page_id,) in query]

    def results_to_qa_preroll(self, limit):
        page_ids_to_ignore = self.get_ignore_page_ids()
        query = self._results_to_qa_query(page_ids_to_ignore)
        query = query.filter(WebPage.preroll_ok == True).limit(limit)
        return [page_id for (page_id,) in query]

    def results_to_qa_non_preroll(self, limit):
        page_ids_to_ignore = self.get_ignore_page_ids()
        query = self._results_to_qa_query(page_ids_to_ignore)
        query = query.filter(WebPage.nonpreroll_ok == True).limit(limit)
        return [page_id for (page_id,) in query]

    def _get_num_hits_submitted(self):
        query = PageHit.query.join(WebPage).filter(PageHit.job_id==self.id)
        preroll_hits = query.filter(WebPage.preroll_ok==True).count()
        non_preroll_hits = query.filter(WebPage.preroll_ok==False).count()
        return preroll_hits, non_preroll_hits

    def results_to_qa(self):
        """ Gather page-ids to be QA'ed for label-id. Page-ids can come from
            both pre-roll and non-pre-roll depending on the preroll_ratio. If
            set to None, both kind of pages are selected randomly.

            Returns:
                list of page_ids
        """
        logger.info("Gathering results for label_id : %s", self.label_id)
        preroll_hits, non_preroll_hits = self._get_num_hits_submitted()
        if self.preroll_ratio is None:
            limit = self._get_limit(self.max_hits_per_submission,
                        preroll_hits + non_preroll_hits, self.max_hits)
            result_set = self.results_to_qa_all(limit)
        else:
            preroll_limit = self._get_limit(
                self.max_hits_per_submission*self.preroll_ratio, preroll_hits,
                self.max_hits*self.preroll_ratio)
            non_preroll_limit = self._get_limit(
                self.max_hits_per_submission*(1 - self.preroll_ratio),
                non_preroll_hits, self.max_hits*(1 - self.preroll_ratio))
            result_set = self.results_to_qa_preroll(preroll_limit) + \
                         self.results_to_qa_non_preroll(non_preroll_limit)
        return result_set

    def submit_hits(self):
        """ Submit screenshot HITs for label to MTurk

            Returns:
                total count of HITs submitted
        """
        evaluator = WebPageTextEvaluator.query.filter_by(
                    target_label_id=self.label_id).one()
        result_set = self.results_to_qa()
        for wp_id in result_set:
            hit_id = evaluator.create_hit(page_id=wp_id)
            vh = PageHit(hit_id=hit_id, label_id=self.label_id, page_id=wp_id,
                        job_id=self.id)
        session.flush()
        return len(result_set)


class MTurkLabelCollageJob(AbstractMTurkLabelJob):
    """ Job class for submitting collage HITs for Labels
    """
    __mapper_args__ = dict(polymorphic_identity='mturk_label_collage_jobs')
    result_table = VideoHit

    def get_ignore_page_ids(self):
        """ Get page-ids that were part of training

            Returns:
                set of page-ids
        """
        training_ids = {ltp.page_id for ltp in LabelTrainingPage.query.
                        filter_by(label_id=self.label_id)}
        return training_ids

    def get_ignore_video_ids(self):
        """ Videos that have been QA'd for the given label
            or an equivalent one

            Returns:
                set of video-ids
        """
        prev_qad_ids = {vh.video_id for vh in VideoHit.query.
                        filter(VideoHit.label_id==self.label_id)}
        return prev_qad_ids

    def results_to_qa(self):
        """ Gather video-id/page-ids to be QA'ed for label-id

            Returns:
                list of tuples of video_id and page_id
        """
        logger.info("Gathering results for label_id : %s", self.label_id)
        label_results = []
        wpi = WebPageInventory
        wplr = WebPageLabelResult

        ignore_video_ids = self.get_ignore_video_ids()
        ignore_page_ids = self.get_ignore_page_ids()
        hits_submitted = self.result_table.query.filter(
                            self.result_table.job_id==self.id).count()
        limit = self._get_limit(self.max_hits_per_submission, hits_submitted,
                                self.max_hits)
        query = session.query(wpi.video_id, wpi.page_id)
        query = query.join(wplr, wplr.page_id == wpi.page_id)
        query = query.distinct(wpi.video_id)

        query = query.filter(wplr.label_id == self.label_id,
                             wpi.video_id != 0)
        if ignore_video_ids:
            query = query.filter(~wpi.video_id.in_(ignore_video_ids))
        if ignore_page_ids:
            query = query.filter(~wpi.page_id.in_(ignore_page_ids))

        query = query.group_by(wpi.video_id)
        query = query.order_by(func.rand()).limit(limit)

        for video_id, page_id in query:
            label_results.append((video_id, page_id))
        return label_results

    def submit_hits(self):
        """ Submit video collage HITs for label to MTurk

            Returns:
                total count of HITs submitted
        """
        evaluator = VideoCollageEvaluator.query.filter_by(
                    target_label_id=self.label_id).one()
        result_set = self.results_to_qa()
        for video_id, wp_id in result_set:
            hit_id = evaluator.create_hit(video_id=video_id)
            vh = VideoHit(hit_id=hit_id, label_id=self.label_id, video_id=video_id,
                page_id=wp_id, job_id=self.id)
        session.flush()
        return len(result_set)

