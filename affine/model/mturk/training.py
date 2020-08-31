from logging import getLogger

from affine.model._sqla_imports import *
from affine.model.web_pages import *
from affine.model.base import Base, session
from affine.model.mturk.evaluators import *
from affine.external.crawl.video_crawler_by_keyword import VideoUrls
from affine.model.labels import Label
from affine.normalize_url import parse_url
from affine.model.mturk.hits import BoxHit, MTurkBox
from affine.vcr.dynamodb import DynamoIngestionStatusClient
from affine.aws import sqs
from affine import config


__all__ = ['VideoTrainingURL', 'TrainingJob']

logger = getLogger(__name__)


class VideoTrainingURL(Base):
    """A URL that should be priority-ingested by the download queue so we have it in our DB"""
    __tablename__ = "video_training_urls"
    id = Column(Integer, primary_key=True)
    training_job_id = Column(Integer,ForeignKey('mturk_training_jobs.id'), nullable=False)
    url = Column(URL, nullable=False)
    processed = Column(Boolean, default=False, nullable=False)

    job = relation('TrainingJob', backref='urls')


class TrainingJob(Base):
    __tablename__ = "mturk_training_jobs"

    id = Column(Integer, primary_key=True)
    evaluator_id = Column(Integer, ForeignKey('mturk_evaluators.id'), nullable=False)
    label_id = Column(Integer, ForeignKey('labels.id'))
    num_urls = Column(Integer, nullable=False)
    search_kw = Column(Unicode(128), nullable=False)
    finished = Column(Boolean, default=False)

    NUM_BOXES_PER_HIT = 21
    youtube_proportion = 50.0
    fivemin_proportion = 25.0
    dailymotion_proportion = 25.0

    label = relation('Label')
    evaluator = relation('MechanicalTurkEvaluator')

    def __unicode__(self):
        s = u'Completed ' if self.finished is True else u''
        return u'%sTraining Job for %s with id: %s and search keyword: %s' % (s, self.label.name, self.id,self.search_kw)

    def create_training_urls(self):
        num_youtube_urls = int(self.youtube_proportion/100 * self.num_urls)
        num_dailymotion_urls = int(self.dailymotion_proportion/100 * self.num_urls)
        num_5min_urls = self.num_urls - num_youtube_urls - num_dailymotion_urls

        urls = set()
        urls.update(VideoUrls.get_youtube_urls(self.search_kw, num_youtube_urls))
        urls.update(VideoUrls.get_5min_urls(self.search_kw, num_5min_urls))
        urls.update(VideoUrls.get_dailymotion_urls(self.search_kw, num_dailymotion_urls))

        dynamo = DynamoIngestionStatusClient()
        download_queue = sqs.get_queue(config.sqs_download_queue_name())
        for url in map(parse_url, urls):
            if dynamo.get(url) is None:
                item_to_enqueue = {"url": url, "status": "Queued", "download_stage": "Text"}
                dynamo.put(item_to_enqueue)
                sqs.write_to_queue(download_queue, item_to_enqueue)
            VideoTrainingURL(job=self, url=url)

        self.num_urls = len(urls)
        session.flush()

        print 'Ingested %s urls to the download queue' %(str(len(urls)))

    def submit_hits(self):
        """Submit facebox hits from the training video url table to MTurk for QA"""
        # query all processed urls (TrainingVideoURL table) for boxes
        # create BoxHits for all boxes and submit hits to Mturk
        boxes = []
        num_hits_submitted = 0
        for url in session.query(VideoTrainingURL).filter_by(training_job_id = self.id, processed = False):
            wpage = WebPage.by_url(url.url)
            if wpage is not None:
                # get the video, and set url.processed only if video is updated on its face version
                videos = sorted(wpage.active_videos, key=lambda x:x.length, reverse = True)
                if len(videos) != 0 :
                    video = videos[0]
                    images_in_s3 = video.s3_timestamps()
                    for b in video.face_boxes:
                        if b.timestamp in images_in_s3:
                            boxes.append(b.id)
                    url.processed = True
                else:
                    url.processed = True
            boxes = sorted(set(boxes))

        for i in xrange(0, len(boxes), self.NUM_BOXES_PER_HIT):
            boxes_per_hit = boxes[i:i+self.NUM_BOXES_PER_HIT]
            hit_id = self.evaluator.create_hit(box_ids=boxes_per_hit)
            session.flush()
            b = BoxHit(hit_id=hit_id, training_job_id=self.id)
            num_hits_submitted += 1
            for box_id in boxes_per_hit:
                if not MTurkBox.query.filter_by(box_id = box_id, label_id = self.label_id).count():
                    MTurkBox(box_id=box_id, hit=b, label_id=self.label_id)
        session.flush()
        return BoxHit, num_hits_submitted

    def _get_job_status(self):
        """ get status of outstanding Hits and unprocessed URLS for a training job """
        total_hits = session.query(BoxHit).filter_by(training_job_id=self.id).count()
        num_hits_left = session.query(BoxHit).filter_by(training_job_id=self.id, outstanding=True).count()
        total_urls = self.num_urls
        num_urls_left = session.query(VideoTrainingURL).filter_by(job=self, processed=False).count()
        faces_obtained = MTurkBox.query.filter_by(label=self.evaluator.target_label, result=True).count()
        return '\n'.join([
            '------------- Stats for Job ID: %s -------------' % str(self.id) ,
            'Job for Label        : %s' % self.label.name,
            'Total URLs           : %d' % total_urls,
            'Total HITs           : %d' % total_hits,
            'unprocessed URLS     : %d' % num_urls_left,
            'outstanding Hits     : %d' % num_hits_left,
            'Job Finish Status    : %s' % self.finished,
            'Faces Obtained       : %d' % faces_obtained,
        ]) + '\n'

    def get_job_status(self):
        print self._get_job_status()

    @classmethod
    def update_status(cls):
        """Ingest new data from MTurk and write it to the database."""
        for job in cls.query.filter(cls.finished == False):
            num_hits_left = session.query(BoxHit).filter_by(training_job_id = job.id, outstanding=True).count()
            urls_left = session.query(VideoTrainingURL).filter_by(training_job_id=job.id, processed = False)
            dynamo = DynamoIngestionStatusClient()
            num_urls_left = 0
            for url in urls_left:
                dynamo_url = dynamo.get(url.url)
                if dynamo_url is None or dynamo_url['status'] == 'Failed':
                    # will never be processed, so ignore for our purposes
                    url.processed = True
                else:
                    num_urls_left += 1
            if num_hits_left+num_urls_left == 0:
                job.finished = True
                print '*** Job ID: %s is complete ***' % str(job.id)

            print '------------- Stats for Job ID: %s -------------' % str(job.id)
            print 'Total URLs      : %i' % VideoTrainingURL.query.filter_by(training_job_id = job.id).count()
            print 'Total HITs      : %i' % BoxHit.query.filter_by(training_job_id = job.id).count()
            if not job.finished:
                print 'unprocessed URLs: %i' % num_urls_left
                print 'outstanding HITs: %i\n' % num_hits_left
        session.flush()

    @classmethod
    def create_with_evaluator_and_training_urls(cls, label_id, num_urls, search_kw, **evaluator_kwargs):
        evluator = ClickableBoxEvaluator.query.filter_by(target_label_id=label_id).first()
        if not evluator:
            evaluator_name = "Training Evaluator for %s" % Label.get(label_id).name
            evluator = ClickableBoxEvaluator(name=evaluator_name, target_label_id=label_id, **evaluator_kwargs)

        training_job = TrainingJob(
            label_id=label_id,
            evaluator=evluator,
            num_urls=num_urls,
            search_kw=search_kw,
        )
        session.flush()
        training_job.create_training_urls()
        return training_job
