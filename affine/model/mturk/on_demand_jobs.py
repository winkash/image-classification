import os
from affine.model.base import *
from affine.model._sqla_imports import *
from affine import config
from affine.aws import s3client
from affine.model.labels import session
from affine.model.mturk.evaluators import MechanicalTurkEvaluator
from logging import getLogger

logger = getLogger(__name__)

EVALUATOR_FOLDER_TYPE = {'images': 'thumbnail', 'videos': 'thumbnail',
                         'page_text':'screenshot'}
NUM_IMAGES_PER_HIT = 21
S3_BASE_URL = 'https://s3.amazonaws.com'

class MTurkOnDemandJob(Base):
    """ On-demand QA for Images and Screenshots which are not necessarily part
        of the inventory.
    """
    __tablename__ = "mturk_on_demand_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(VARCHAR(128), nullable=False)
    created = Column(Timestamp, nullable=False, server_default=func.now())
    resource_id = Column(Integer, nullable=False)
    resource_url = Column(URL, nullable=False)
    resource_name = Column(VARCHAR(128), default=None)
    evaluator_id = Column(Integer, ForeignKey('mturk_evaluators.id'))
    hit_id = Column(VARCHAR(128), nullable=False)
    result = Column(Boolean, default=None)
    outstanding = Column(Boolean, nullable=False, default=True)

    __table_args__ = (UniqueConstraint('job_id', 'resource_id',
        name='modj_jobid_resid'), {})

    def __unicode__(self):
        return u'MTurkOnDemandJob job_id:%s, resource_id:%s, hit_id:%s' % (
            self.job_id, self.resource_id, self.hit_id)

    @classmethod
    def submit_hits(cls, evaluator_id, image_folder_path, job_id):
        """ Submit On-demand HITs to MTurk.
            Function mocks input evaluator when not in prod

            Args:
                evaluator_id: evaluator to be used for the job
                image_folder_path: path to local directory with images to be QA'ed
                job_id: name to uniquely identify the job (e.g. logo-prod_2015-01-01)
        """
        evaluator = MechanicalTurkEvaluator.get(evaluator_id)
        if 'sandbox' in config.get("mturk_hostname"):
            evaluator.mock_evaluator = True
        ev_type = evaluator.evaluator_type
        msg = "The selected evaluator is not currently supported"
        assert ev_type in EVALUATOR_FOLDER_TYPE, msg
        logger.info('Uploading to S3...')
        s3_urls = cls._upload_to_s3(image_folder_path, ev_type, job_id)
        logger.info('Creating HITs...')
        if ev_type == 'images':
            cls._submit_image_hits(evaluator, s3_urls, job_id)
        elif ev_type == 'page_text':
            cls._submit_screenshot_hits(evaluator, s3_urls, job_id)
        elif ev_type == 'videos':
            cls._submit_collage_hits(evaluator, s3_urls, job_id)

    @classmethod
    def _submit_image_hits(cls, evaluator, s3_urls, job_id):
        """ Function to create HITs which use thumbnails
            Creates entries in MTurkOnDemandJob table for each input resource

            Args:
                evaluator: input evaluator
                s3_urls: dict of the form - {(job_id, resource_id):s3_url}
                job_id: name of the job
        """
        for url_subset in cls._chunks(sorted(s3_urls.keys()), NUM_IMAGES_PER_HIT):
            hit_id = evaluator.create_hit(is_on_demand=True, image_ids=url_subset)
            for j_id, res_id, res_name in url_subset:
                cls(job_id=job_id, hit_id=hit_id, resource_url=s3_urls[(j_id, res_id, res_name)],
                    resource_id=res_id, resource_name=res_name, evaluator_id=evaluator.id)
        session.flush()

    @classmethod
    def _submit_screenshot_hits(cls, evaluator, s3_urls, job_id):
        """ Function to create HITs which use screenshot
            Creates entries in MTurkOnDemandJob table for each input resource

            Args:
                evaluator: input evaluator
                s3_urls: dict of the form - {(job_id, resource_id):s3_url}
                job_id: name of the job
        """
        for (j_id, res_id, res_name), url in s3_urls.iteritems():
            hit_id = evaluator.create_hit(is_on_demand=True,
                image_id=[j_id, res_id])
            cls(job_id=job_id, hit_id=hit_id, resource_url=url,
                resource_id=res_id, resource_name=res_name, evaluator_id=evaluator.id)
        session.flush()

    @classmethod
    def _submit_collage_hits(cls, evaluator, s3_urls, job_id):
        """ Function to create HITs which use collages
            Creates entries in MTurkOnDemandJob table for each input resource
            (in this case a folder containing the collage of thumbnails)

            Args:
                evaluator: input evaluator
                s3_urls: dict of the form - {(job_id, resource_id):s3_url}
                job_id: name of the job
        """
        for (j_id, res_id, res_name), url in s3_urls.iteritems():
            hit_id = evaluator.create_hit(folder_id=[j_id, res_id])
            cls(job_id=job_id, hit_id=hit_id, resource_url=url,
                resource_id=res_id, resource_name=res_name, evaluator_id=evaluator.id)
        session.flush()

    @staticmethod
    def _chunks(l, n):
        """ Yield successive n-sized chunks from input list l """
        for i in xrange(0, len(l), n):
            yield l[i:i+n]

    @classmethod
    def _upload_to_s3(cls, image_folder_path, evaluator_type, job_id):
        """ Upload images from local directory to S3
            Images are uploaded to s3 to the on-demand QA bucket in either thumbnail
            or screenshot folder depending on the evaluator-type. Each job-id has
            a unique folder. Images are renamed to their resource-id numbers.

            Args:
                image_folder_path: path to local directory with images to be QA'ed
                evaluator_type: either images or page-text
                job_id: name of the job
        """
        bucket = config.s3_on_demand_qa_bucket()
        ev_folder = EVALUATOR_FOLDER_TYPE[evaluator_type]
        resource_id = 1
        s3_urls = {}
        for res_name in filter(lambda f: not f.startswith('.'), os.listdir(image_folder_path)):
            url_path = '%s/%s/%s' %(ev_folder, job_id, resource_id)
            local_path = os.path.join(image_folder_path, res_name)
            if evaluator_type == 'videos':
                s3client.upload_folder_to_s3(bucket, url_path, local_path, True)
            else:
                s3client.upload_to_s3(bucket, url_path, local_path, True)
            s3_urls[(job_id, resource_id, res_name)] = '%s/%s/%s' %(S3_BASE_URL,
                bucket, url_path)
            resource_id += 1
        return s3_urls


    @classmethod
    def download_results(cls, job_id, local_dir, result):
        """ Download results locally

            Args:
                job_id: input job-id
                local_dir: Path to local directory where results are to be downloaded
                result (Boolean): True/False results to be downloaded
        """
        ev_id = cls.query.filter_by(job_id=job_id).first().evaluator_id
        ev_type = MechanicalTurkEvaluator.get(ev_id).evaluator_type
        ev_folder = EVALUATOR_FOLDER_TYPE[ev_type]
        if ev_type == 'videos':
            cls._download_folder(ev_folder, job_id, local_dir, result)
        else:
            cls._download_files(ev_folder, job_id, local_dir, result)

    @classmethod
    def _download_files(cls, ev_folder, job_id, local_dir, result):
        conn = s3client.connect(config.s3_on_demand_qa_bucket())
        for res in cls.query.filter_by(job_id=job_id, result=result):
            url_path = "%s/%s/%s" %(ev_folder, job_id, res.resource_id)
            local_path = os.path.join(local_dir, res.resource_name)
            conn.get_key(url_path).get_contents_to_filename(local_path)

    @classmethod
    def _download_folder(cls, ev_folder, job_id, local_dir, result):
        conn = s3client.connect(config.s3_on_demand_qa_bucket())
        for res in cls.query.filter_by(job_id=job_id, result=result):
            thumbnails = cls.get_thumbnails_for_collage(job_id, res.resource_id)
            local_res_dir = os.path.join(local_dir, res.resource_name)
            if not os.path.exists(local_res_dir):
                os.makedirs(local_res_dir)
            for thumb in thumbnails:
                url_path = "%s/%s/%s/%s" %(ev_folder, job_id, res.resource_id, thumb)
                local_res_file = os.path.join(local_res_dir, thumb)
                conn.get_key(url_path).get_contents_to_filename(local_res_file)

    @classmethod
    def get_job_status(cls, job_id):
        """ Get status on job progress
        """
        total_urls = cls.query.filter_by(job_id=job_id).count()
        num_urls_left = cls.query.filter_by(job_id=job_id, outstanding=True).count()
        num_trues = cls.query.filter_by(job_id=job_id, result=True).count()
        percent_done = 100*float(total_urls - num_urls_left)/total_urls if total_urls else 0
        odj = cls.query.filter_by(job_id=job_id).first()
        ev = MechanicalTurkEvaluator.get(odj.evaluator_id) if odj is not None else None
        result = '\n'.join([
            '------------- Stats for Job ID: %s -------------' %job_id,
            'Evaluator              : %s' % ev,
            'Total resources in job : %d' % total_urls,
            'Unprocessed resources  : %d' % num_urls_left,
            '%% complete            : %f' % percent_done,
            'Num Trues Obtained     : %d' % num_trues,
        ]) + '\n'
        return result

    @classmethod
    def get_thumbnails_for_collage(cls, job_id, resource_id, limit=None):
        bucket = config.s3_on_demand_qa_bucket()
        key = s3client.connect(bucket)
        ev_folder = EVALUATOR_FOLDER_TYPE['videos']
        url_path = "%s/%s/%s/" %(ev_folder, job_id, resource_id)
        return [f.name.split('/')[-1] for f in key.list(prefix=url_path)][:limit]

