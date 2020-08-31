from affine.model.base import *
from affine.model._sqla_imports import *
from affine import config
from affine.model.labels import Label
from affine.model.mturk.evaluators import ClickableImageEvaluator, \
    VideoCollageEvaluator
from affine.model.mturk.hits import MTurkImage, ImageHit, VideoHit
from affine.model.mturk.evaluators import get_hits_by_evaluator
from logging import getLogger

__all__ = ['MTurkImageJob']

logger = getLogger(__name__)

EVALUATOR_TYPES_SUPPORTED = [ClickableImageEvaluator, VideoCollageEvaluator]


class MTurkImageJob(Base):
    __tablename__ = "mturk_image_jobs"
    id = Column(Integer, primary_key=True)
    label_id = Column(Integer, ForeignKey('labels.id'))
    finished = Column(Boolean, default=False)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    NUM_IMAGES_PER_HIT = 21

    def __init__(self, label_id, evaluator_type=ClickableImageEvaluator,
                 **kwargs):
        super(MTurkImageJob, self).__init__()

        assert (evaluator_type in EVALUATOR_TYPES_SUPPORTED), \
            "evaluator_type has to be in %s" % str(EVALUATOR_TYPES_SUPPORTED)

        self.label_id = label_id
        self.finished = False
        self.evaluator_type = evaluator_type
        self.hit = evaluator_type.hit_type

        if evaluator_type.query.filter_by(target_label_id=label_id).count():
            evaluator = evaluator_type.query.filter_by(
                target_label_id=label_id).one()
        else:
            evaluator_name = "MTurk Image Evaluator for %s" % Label.get(
                label_id).name
            evaluator = evaluator_type(
                name=evaluator_name, target_label_id=label_id, **kwargs)
            session.flush()

        # only for stage and dev
        if 'sandbox' in config.get("mturk_hostname"):
            evaluator.min_hits_approved = 0
            session.flush()

    def __unicode__(self):
        return u'<MTurkImageJob label: %s, Completion status: %s>' % \
            (Label.get(self.label_id).name, str(self.finished))

    def _submit_image_hits(self, evaluator, images):

        num_hits_submitted = 0
        for i in xrange(0, len(images), self.NUM_IMAGES_PER_HIT):
            images_per_hit = images[i:i + self.NUM_IMAGES_PER_HIT]
            hit_id = evaluator.create_hit(image_ids=images_per_hit)
            ih = ImageHit(hit_id=hit_id, mturk_image_job_id=self.id)
            num_hits_submitted += 1
            session.flush()
            for video_id, ts in images_per_hit:
                if not MTurkImage.query.filter_by(video_id=video_id,
                                                  timestamp=ts,
                                                  label_id=self.label_id).count():
                    MTurkImage(video_id=video_id, timestamp=ts,
                               label_id=self.label_id, image_hit_id=ih.id)
            session.flush()

        return ImageHit, num_hits_submitted

    def _submit_vc_hits(self, evaluator, video_ids):
        num_hits_submitted = 0
        video_ids = list(set(video_ids))
        for vid in video_ids:
            if not VideoHit.query.filter_by(video_id=vid,
                                            label_id=self.label_id).count():
                hit_id = evaluator.create_hit(video_id=vid)
                VideoHit(hit_id=hit_id, label_id=self.label_id, video_id=vid)
                num_hits_submitted += 1
        session.flush()

        return VideoHit, num_hits_submitted

    def submit_hits(self, data):
        """
        Submit mturk hits with the input data.

        Args:
            data:
              if self.evaluator_type is ClickableImageEvaluator, 
                data going to create_hit (submit_image_hits) has to be 
                list of lists [video-id,image-timestamp]
              if self.evaluator_type is VideoCollageEvaluator, 
                data going to create_hit (submit_vc_hits) has to be list of video ids
        Returns:
            hit_type, number of hits submitted
        """
        evaluator = self.evaluator_type.query.filter_by(
            target_label_id=self.label_id).one()

        num_hits_submitted = 0
        hit_type = None
        if 'sandbox' in config.get("mturk_hostname"):
            evaluator.min_hits_approved = 0
            session.flush()
            if type(evaluator) == ClickableImageEvaluator:
                assert (len(data) > 0 and len(data[0]) == 2), \
                    "If we use ClickableImageEvaluator hits, data needs to be a list of lists [video-id,image-timestamp]"
                hit_type, num_hits_submitted = self._submit_image_hits(
                    evaluator, data)
            else:
                assert (len(data) > 0 and type(data[0]) is not list), \
                    "If we use VideoCollageEvaluator hits, data needs to be a list of video ids"
                hit_type, num_hits_submitted = self._submit_vc_hits(
                    evaluator, data)
        return hit_type, num_hits_submitted

    @classmethod
    def ingest_results(cls):
        """Ingest new data from MTurk Image Hits and write it to text files"""
        logger.info("Ingesting Results for ImageHits from MTurkImageJob")
        for evaluator_id, compdict in get_hits_by_evaluator().iteritems():
            evaluator = ClickableImageEvaluator.get(evaluator_id)
            if evaluator is not None:
                for hit_id, mt_result in compdict.iteritems():
                    evaluator.ingest_hit(hit_id, mt_result)

    def update_status(self):
        """ 
        Update the job status (self.finished=True) if the there is no 
        outstanding ImageHit with current job id (self.id)
        """
        incomplete_hits = ImageHit.query.filter_by(
            mturk_image_job_id=self.id, outstanding=True).count()
        if incomplete_hits == 0:
            self.finished = True
            session.flush()

        return incomplete_hits

    def update_status_from_vids(self, video_ids):
        """ 
        Update the job status (self.finished=True) if the there is no outstanding 
        VideoHit the job target label (self.label_id) and any of the input video ids
        """
        incomplete_hits = VideoHit.query.\
            filter(VideoHit.label_id == self.label_id,
                   VideoHit.video_id.in_(video_ids),
                   VideoHit.outstanding == True).count()
        if incomplete_hits == 0:
            self.finished = True
            session.flush()
