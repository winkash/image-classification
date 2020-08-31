import simplejson as json
from unidecode import unidecode
import unicodedata
import sys
import re
import traceback
from collections import defaultdict
from logging import getLogger
from math import ceil
from boto.mturk.connection import MTurkRequestError
from affine import config
from affine.aws.mturk import MTurkUtils
from affine.model._sqla_imports import *
from affine.model.base import session, Base
from affine.model.labels import Label
from affine.model.videos import Video
from affine.model.boxes import Box
from affine.model.web_pages import WebPage
from affine.model.mturk.hit_failures import MTurkHitFailure
from affine.model.mturk.workers import MTurkWorker
from affine.model.mturk.hits import VideoHit, GoldenHit, PageHit, MTurkImage, \
    MTurkBox, ImageHit, BoxHit
from affine.video_processing import sample_items


__all__ = ['MechanicalTurkEvaluator', 'ClickableBoxEvaluator',
           'VideoCollageEvaluator', 'ClickableImageEvaluator',
           'WebPageTextEvaluator']

logger = getLogger(__name__)


class MechanicalTurkEvaluator(Base):

    """A question that we can ask to MTurk to get info about a video/ image/ page.

    MechanicalTurkEvaluator is an abstract base class. Each child class
    represents a template for a type of MechanicalTurkEvaluator job.
    There is one instances of each subclass per label that we have
    asked about for that template.
    """
    __tablename__ = "mturk_evaluators"

    id = Column(Integer, primary_key=True)
    name = Column(Unicode(128), nullable=False)
    super = Column(Boolean, nullable=False, default=False)
    _cls = Column('cls', String(50), nullable=False)
    __mapper_args__ = dict(polymorphic_on=_cls)

    question = Column(UnicodeText, nullable=False)
    reference_image_url = Column(URL, nullable=True)
    title = Column(UnicodeText, nullable=False)
    description = Column(UnicodeText, nullable=False)
    keywords = Column(UnicodeText, nullable=False)
    approval_delay = Column(Integer, default=604800, nullable=False)
    reward_amt = Column(Integer, default=1, nullable=False)  # In cents
    lifetime = Column(Integer, default=172800, nullable=False)
    duration = Column(Integer, default=604800, nullable=False)
    # Worked qualification criteria
    min_percent_approved = Column(Integer, default=98)
    require_adult = Column(Boolean, default=False, nullable=False)
    max_assignments = Column(Integer, default=4, nullable=False)
    # number of yeses needed out of max_assignments total
    match_threshold = Column(Integer, default=3, nullable=True)

    target_label_id = Column(Integer, ForeignKey('labels.id'))
    target_label = relation('Label', backref="mturk_evaluators")

    # this default value is being tested and may be added to the DB later
    min_hits_approved = 5000
    # set this value to true to test HITs in MTurk sandbox using mock_evaluator
    __mock_evaluator = False

    @property
    def mock_evaluator(self):
        return self.__mock_evaluator

    @mock_evaluator.setter
    def mock_evaluator(self, value):
        message = "Cannot mock evaluator in prod"
        assert 'sandbox' in config.get('mturk_hostname'), message
        message = "Input value should be boolean"
        assert isinstance(value, bool), message
        self.__mock_evaluator = value

    def mock_evaluator_for_testing(func):
        def set_values(self, **kwargs):
            bucket = config.get('affine.s3.bucket')
            try:
                if self.mock_evaluator:
                    self.min_percent_approved = 0
                    self.max_assignments = 1
                    self.min_hits_approved = 0
                    self.match_threshold = 1
                    self.require_adult = False
                    session.flush()
                    config.set('affine.s3.bucket', 'affine')
                f = func(self, **kwargs)
                return f
            finally:
                config.set('affine.s3.bucket', bucket)
        return set_values

    true_answer = ["yes"]
    false_answer = ["no"]

    @classmethod
    def get_s3_bucket(cls, is_on_demand):
        """ Returns the correct S3 bucket depending on on-demand or regular HIT """
        return config.get('affine.s3.on_demand_qa_bucket') if is_on_demand else config.get('affine.s3.bucket')

    @classmethod
    def get_or_create(cls, target_label):
        existing = cls.query.filter_by(target_label=target_label).first()
        if existing:
            return existing
        evaluator = cls(name=target_label.name, target_label=target_label)
        session.flush()
        return evaluator

    def get_result(self, golden_hit_id):
        """Each subclass has to define how to get the hit's result for its
        golden_hit"""
        raise NotImplementedError

    @staticmethod
    def extract_hit_data(hit):
        """Each subclass has to define how to extract data from hits it uses"""
        raise NotImplementedError

    def generate_html(self, **template_data):
        return MTurkUtils.render_mturk_question_template(template_name=self.template, **template_data)

    @staticmethod
    def create_duplicate_hit(hit):
        """Submit a new hit identical to an existing one."""
        # evaluator is NOT mocked for testing
        cls = HIT_TYPE_TO_EVALUATOR_TYPE[type(hit)]
        data, target_label = cls.extract_hit_data(hit)
        evaluator = cls.query.filter_by(target_label=target_label).one()
        return evaluator.create_hit(**data)

    @mock_evaluator_for_testing
    def create_hit(self, **kwargs):
        """Submit a task to MTurk"""
        reward_amt = self.reward_amt / 100.0
        try:
            try:
                template_data = self.format_data(**kwargs)
                hit_html = self.generate_html(**template_data)
                hit_id = MTurkUtils.submit_hit(hit_html, self.title, self.description, self.keywords, self.approval_delay,
                    reward_amt, self.duration, self.lifetime, self.max_assignments, require_adult=self.require_adult,
                    min_percent_approved=self.min_percent_approved, min_hits_approved=self.min_hits_approved, require_us=True)
            except (MTurkRequestError, UnicodeEncodeError):
                if self.evaluator_type == 'page_text':
                    template_data = self.format_data(process_title=True, **kwargs)
                    hit_html = self.generate_html(**template_data)
                    hit_id = MTurkUtils.submit_hit(hit_html, self.title, self.description, self.keywords, self.approval_delay,
                        reward_amt, self.duration, self.lifetime, self.max_assignments, require_adult=self.require_adult,
                        min_percent_approved=self.min_percent_approved, min_hits_approved=self.min_hits_approved, require_us=True)
                else:
                    raise
        except Exception:
            logger.info('HIT creation failed for %s' %kwargs)
            tb = traceback.format_exc() + '\n input kwargs: %s' %kwargs
            if 'AWS.MechanicalTurk.InsufficientFunds' not in tb:
                MTurkHitFailure(hit_id='Invalid HIT', message=tb)
                session.flush()
            raise

        logger.info('created %s' % hit_id)
        return hit_id

    def ingest_golden_hit(self, hit_id, assignments):
        ''' Updates num_golden_error, num_golden columns of worker table '''
        msg = 'processing golden-hit %s of type %s' %(hit_id, self.hit_type.__name__)
        logger.info(msg)
        sub_hits = self.extract_sub_hits(hit_id, assignments)
        for hit_id, assignments in sub_hits.iteritems():
            for assignment in assignments:
                worker_id, answer = assignment[
                    'worker_id'], assignment['answer']
                worker = MTurkWorker.get_or_create(worker_id)
                answer = answer == self.true_answer
                worker.num_golden += 1
                worker.num_golden_error += answer != self.get_result(hit_id)

    def ingest_on_demand_hit(self, hit_id, assignments):
        raise NotImplementedError

    def ingest_hit(self, hit_id, assignments):
        from affine.model.mturk import MTurkOnDemandJob
        with session.begin():
            if GoldenHit.query.filter_by(golden_hit_id=hit_id).count():
                self.ingest_golden_hit(hit_id, assignments)
            elif MTurkOnDemandJob.query.filter_by(hit_id=hit_id).count():
                processed_hit = self.ingest_on_demand_hit(hit_id, assignments)
                self.update_on_demand_job_status(processed_hit)
                self.update_workers(hit_id, assignments)
            else:
                processed_hit = self.process_hit(hit_id, assignments)
                self.hit_type.update_mturk_results(processed_hit)
                self.update_workers(hit_id, assignments)
            MTurkUtils.delete_hit(hit_id)

    def save_video_result(self, video_id, result):
        from affine.model.mturk.results import MTurkVideoResult
        MTurkVideoResult.set_result(video_id, self.id, result)

    def update_workers(self, hit_id, assignments):
        ''' Updates worker db with yes,no,time but NOT minority or conflict '''
        sub_hits = self.extract_sub_hits(hit_id, assignments)
        try:
            for hit_id, assignments in sub_hits.iteritems():
                worker_set = WorkerSet(hit_id)
                for assignment in assignments:
                    answer, worker_id, time_elapsed = assignment['answer'], \
                        assignment['worker_id'], assignment['time_elapsed']
                    worker = MTurkWorker.get_or_create(worker_id)
                    worker.time_elapsed += time_elapsed
                    if answer == self.true_answer:
                        worker.yes_count += 1
                    elif answer == self.false_answer:
                        worker.no_count += 1
                    else:
                        raise Exception('Worker response parse error', answer)
                    worker_set.add_worker(worker_id, worker,
                                          answer ==
                                          self.true_answer)
                # Update workerset with minority information
                worker_set.update_workers()
        except Exception:
            logger.exception('Failed to process worker info')
            raise

    def update_on_demand_job_status(self, mt_results):
        from affine.model.mturk import MTurkOnDemandJob
        for hit_id, job_id, resource_id, label_id, result in mt_results:
            mj = MTurkOnDemandJob.query.filter_by(resource_id=resource_id,
                job_id=job_id, hit_id=hit_id).first()
            if not mj:
                msg = "MTurkOnDemandJob not found for hit_id:%s, job_id:%s, thumbnail:%s"
                logger.warn(msg %(hit_id, job_id, resource_id))
            else:
                mj.result = result
                mj.outstanding = False
        session.flush()

    def extract_sub_hits(self, hit_id, assignments):
        return {hit_id: assignments}

    @classmethod
    def _split_image_id(cls, image_id_str):
        try:
            folder_id, thumb = map(int, image_id_str.split('_'))
        except ValueError:
            i = image_id_str.rfind('_')
            folder_id, thumb = image_id_str[:i], image_id_str[i+1:]
            thumb = int(thumb)
        return folder_id, thumb


class VideoCollageEvaluator(MechanicalTurkEvaluator):

    """An MTurk template where we show a series of images from a video and ask a yes/ no question.

    For example, "Does this video have soccer content?"
    """
    __mapper_args__ = dict(polymorphic_identity='video_collage')

    template = "video_collage_question_form.html"
    default_keywords = "Categorization,Videos,Tag,Label,Keyword,Image,Photo"
    evaluator_type = "videos"
    hit_type = VideoHit
    NUM_FRAMES_PER_HIT = 16

    def __init__(self, **kwargs):
        super(VideoCollageEvaluator, self).__init__(**kwargs)
        label = self.target_label
        if label is None and self.target_label_id is not None:
            label = Label.get(self.target_label_id)
        if label is not None:
            if self.question is None:
                self.question = 'Does this video contain %s content?' % label.name
            if self.title is None:
                self.title = 'Image Categorization (%s)' % label.name
            if self.description is None:
                self.description = ('You will be shown a series of images from a single video and asked '
                                    'whether the video contains %s content' % label.name)
            # we found an instance of a porn video being sent as a HIT for other labels
            # so we must mark all our video collage hits as requiring adult to forestall
            # disciplinary action from Amazon -- we'll find a better solution
            # eventually.
            self.require_adult = True

        if self.keywords is None:
            self.keywords = self.default_keywords

    @staticmethod
    def extract_hit_data(hit):
        return {"video_id": hit.video_id}, hit.label

    def s3_box_timestamps(self, video, limit=None):
        boxes = video.face_boxes
        box_ts = {box.timestamp for box in boxes}
        len_box_ts = len(box_ts)
        result_ts = []
        if not box_ts:
            result_ts = video.s3_timestamps()
            result_ts = sample_items(result_ts, num_items=limit)
        elif limit is None or len_box_ts >= limit:
            result_ts = sample_items(list(box_ts), num_items=limit, border=0)
        elif len_box_ts < limit:
            ts = set(video.s3_timestamps())
            filter_ts = ts - box_ts
            filter_ts = sample_items(filter_ts, num_items=limit-len_box_ts)
            result_ts = box_ts.union(filter_ts)
        return sorted(list(result_ts))

    def format_data(self, video_id=None, folder_id=None):
        from affine.model.mturk import MTurkOnDemandJob
        msg = 'Function must be called either with video_id OR folder_id argument'
        assert (video_id or folder_id) and not (video_id and folder_id), msg
        if folder_id is not None:
            msg = 'folder_id must be a list of [job_id, resource_id]'
            assert type(folder_id) is list and len(folder_id) == 2, msg

        data = {
            'question': self.question,
            'image_bucket': self.get_s3_bucket(is_on_demand=True if folder_id else False),
        }
        data['data'] = {
            "clickable": 'false',
            "labels": ["yes", "no"],
            "evaluator_id": str(self.id),
        }

        if video_id:
            video = Video.get(video_id)
            if self.target_label.label_type=='personality':
                thumbnails = self.s3_box_timestamps(video, self.NUM_FRAMES_PER_HIT)
            else:
                thumbnails = video.s3_timestamps()
                thumbnails = sample_items(thumbnails, num_items=self.NUM_FRAMES_PER_HIT)
            thumbnails = map(str, thumbnails)
            data['data'].update({"video_id": str(video_id),
                "thumbnails": thumbnails})
        elif folder_id:
            job_id, resource_id = folder_id
            thumbnails = MTurkOnDemandJob.get_thumbnails_for_collage(job_id,
                resource_id, self.NUM_FRAMES_PER_HIT)
            thumbnails = map(str, thumbnails)
            data['data'].update({"folder_id":folder_id,
                "thumbnails": thumbnails})
        return data

    def get_result(self, golden_hit_id):
        result = session.query(VideoHit.result).join(GoldenHit,
                                                     GoldenHit.hit_id == VideoHit.hit_id).filter(
            GoldenHit.golden_hit_id == golden_hit_id).\
            scalar()
        return result

    def process_hit(self, hit_id, assignments):
        processed_results = []
        logger.info('processing video hit %s' % hit_id)
        # try:
        num_yes = sum((assignment['answer'] ==
                       self.true_answer)
                      for assignment in assignments)
        num_nos = len(assignments) - num_yes
        video_id = int(assignments[0]['video_id'][0])
        if num_yes >= self.match_threshold:
            result = True
            self.save_video_result(video_id, result)
        elif num_nos >= self.match_threshold:
            result = False
            self.save_video_result(video_id, result)
        else:
            result = None
        processed_results.append([hit_id, video_id, self.target_label_id,
                                  result])
        # except:
        return processed_results

    def ingest_on_demand_hit(self, hit_id, assignments):
        processed_results = []
        logger.info('processing on-demand video hit %s' % hit_id)
        num_yes = sum((assignment['answer'] ==
                       self.true_answer)
                      for assignment in assignments)
        num_nos = len(assignments) - num_yes
        folder_id = assignments[0]['folder_id'][0]
        job_id, resource_id = self._split_image_id(folder_id)
        if num_yes >= self.match_threshold:
            result = True
        elif num_nos >= self.match_threshold:
            result = False
        else:
            result = None
        processed_results.append([hit_id, job_id, resource_id, self.target_label_id,
                                  result])
        return processed_results


class ClickableBoxEvaluator(MechanicalTurkEvaluator):

    """ An MTurk template where we show a series of images that have bounding boxes around boxes and ask whether the faces are of a given person.
        eg: "Click on the images where Tom Cruise's face is contained by the red box.
    """
    __mapper_args__ = dict(polymorphic_identity='clickable_box')

    template = "clickable_box_detection.html"
    default_keywords = "Categorization,Videos,Tag,Label,Keyword,Image,Photo,Celebrity,Clickable"
    evaluator_type = "boxes"
    box_pat = re.compile('^box_[0-9]+$')
    hit_type = MTurkBox

    def __init__(self, **kwargs):
        super(ClickableBoxEvaluator, self).__init__(**kwargs)
        label = self.target_label
        if label is None and self.target_label_id is not None:
            label = Label.get(self.target_label_id)
        if label is not None:
            if self.question is None:
                self.question = "Click on the images where %s's face is contained by the red box." % label.name
            if self.title is None:
                self.title = 'Clickable Image Tagging (%s)' % label.name
            if self.description is None:
                self.description = (
                    'You will be shown a series of images and asked to click the ones that have %s enclosed in a red box' % label.name)
        if self.keywords is None:
            self.keywords = self.default_keywords

    @staticmethod
    def extract_hit_data(hit):
        return {"box_ids": [mb.box_id for mb in hit.boxes]}, hit.boxes[0].label

    def format_data(self, box_ids):
        """ideal number of box ids is 18, but the template will not break if there are more or less"""
        boxes = [Box.get(face_id) for face_id in box_ids]
        videos = defaultdict(dict)
        for box in boxes:
            video_id = int(box.video_id)
            box_id = int(box.id)
            videos[box_id]['video_id'] = video_id
            videos[box_id]['thumbnail'] = map(
                int, [box.timestamp, box.x, box.y, box.width, box.height])
        data = {
            'question': self.question,
            'image_bucket': config.get('affine.s3.bucket'),
        }
        data['data'] = {
            "evaluator_id": str(self.id),
            "videos": dict(videos),
            "reference_image": self.reference_image_url if self.reference_image_url is not None else '',
        }
        return data

    def process_hit(self, hit_id, assignments):
        processed_results = []
        logger.info('processing box hit %s' % hit_id)
        # get a list of the box ids for this hit from the first assignment in
        # the assignments
        box_ids = assignments[0]['box_ids'][0].split('_')
        # combine all the assignments' results into a single list
        all_clicked_boxes = reduce(
            lambda x, y: list(set(x) | set(y.keys())), assignments)
        all_clicked_boxes = [
            k.replace('box_', '') for k in all_clicked_boxes if self.box_pat.search(k)]

        # guaranteed 3/3 if nobody clicked
        false_box_ids = set(box_ids) - set(all_clicked_boxes)
        for box_id in false_box_ids:
            processed_results.append(
                (hit_id, int(box_id), self.target_label_id, False))
        true_results = defaultdict(int)
        for assignment in assignments:
            for box_id in all_clicked_boxes:
                munged = "box_%s" % str(box_id)
                true_results[box_id] += 1 if munged in assignment else 0
        true_box_ids = set()
        for box_id, num_true in true_results.iteritems():
            num_false = self.max_assignments - num_true
            if num_true >= self.match_threshold:
                result = True
                true_box_ids.add(box_id)
            elif num_false >= self.match_threshold:
                result = False
            else:
                result = None
            processed_results.append(
                (hit_id, int(box_id), self.target_label_id, result))
        # update video results if true face boxes are found
        if true_box_ids:
            query = session.query(Box.video_id.distinct()).filter(
                Box.id.in_(true_box_ids),
                or_(Box.box_type == 'Face', Box.box_type == 'Logo'))
            for (vid,) in query:
                self.save_video_result(vid, True)
        return processed_results

    def get_result(self, golden_hit_id):
        # We need the result for the MTurkBox corresponding to golden_hit_id
        # Hence we get the MTurkBox with box_id where corresponding BoxHit with
        # hit_id has a GoldenHit with golden_hit_id
        golden_hit_id, box_id = golden_hit_id.split('_')
        result = \
            session.query(MTurkBox.result).join(MTurkBox.hit).join(GoldenHit,
                                                                   GoldenHit.hit_id ==
                                                                   BoxHit.hit_id).filter(GoldenHit.golden_hit_id == golden_hit_id,
                                                                                         MTurkBox.box_id == box_id).scalar()
        return result

    def extract_sub_hits(self, hit_id, assignments):
        sub_hits = defaultdict(list)
        for assignment in assignments:
            time_elapsed, worker_id = assignment['time_elapsed'], \
                assignment['worker_id']
            item_ids = assignment['box_ids'][0].split('_')
            all_clicked_items = [k.replace('box_', '')
                                 for k in assignment
                                 if self.box_pat.search(k)]
            false_item_ids = set(item_ids) - set(all_clicked_items)
            items = [(self.false_answer, false_id)
                     for false_id in false_item_ids]
            items += [(self.true_answer,
                       click_id) for click_id in all_clicked_items]
            time_elapsed = int(ceil(time_elapsed / float(len(item_ids))))
            for response, item_id in items:
                sub_hit_id = hit_id + '_' + item_id
                sub_hits[sub_hit_id].append({
                    'worker_id': worker_id,
                    'time_elapsed': time_elapsed,
                    'answer': response,
                })
        return sub_hits


class ClickableImageEvaluator(MechanicalTurkEvaluator):

    """ An MTurk Template where we show a series of images and ask whether the
    image contents are of a particular category.
    This will be used primarily to create new BOVW detectors.
        eg: Click on the images where image contents are related to Soccer?
    """
    __mapper_args__ = dict(polymorphic_identity='clickable_image')

    template = 'clickable_image_detection.html'
    default_keywords = "Categorization,Videos,Tag,Label,Keyword,Image,Photo,Celebrity,Clickable"
    evaluator_type = "images"
    image_pat = re.compile('^image_[\x00-\x7F]+_[0-9]+$')
    hit_type = MTurkImage

    def __init__(self, **kwargs):
        super(ClickableImageEvaluator, self).__init__(**kwargs)
        label = self.target_label
        if label is None and self.target_label_id is not None:
            label = Label.get(self.target_label_id)
        if label is not None:
            if self.question is None:
                self.question = "Click on the images whose content is related to %s. " % label.name
            if self.title is None:
                self.title = 'Clickable Image Tagging (%s)' % label.name
            if self.description is None:
                self.description = (
                    'You will be shown a series of images and asked to click the ones whose content is related to %s ' % label.name)
        if self.keywords is None:
            self.keywords = self.default_keywords

    @staticmethod
    def extract_hit_data(hit):
        return {"image_ids": [[mi.video_id, mi.timestamp] for mi in hit.images]},\
            hit.images[0].label

    def format_data(self, image_ids, is_on_demand=False):
        bucket = self.get_s3_bucket(is_on_demand=is_on_demand)
        data = {
            'question': self.question,
            'image_bucket': bucket,
        }
        data['data'] = json.dumps({
            "evaluator_id": str(self.id),
            "image_ids": image_ids,
        })
        return data

    def get_result(self, golden_hit_id):
        golden_hit_id, video_id, timestamp = golden_hit_id.split('_')
        # We need the MTurkImage's result where,
        # The GoldenHit with golden_hit_id has a hit_id corresponding to an
        # ImageHit which in turn has the required MTurkImage with video_id and
        # timestamp
        result = session.query(MTurkImage.result).\
            join(MTurkImage.hit).join(GoldenHit,
                                      GoldenHit.hit_id == ImageHit.hit_id).filter(
            GoldenHit.golden_hit_id == golden_hit_id,
            MTurkImage.timestamp == timestamp,
            MTurkImage.video_id == video_id).scalar()
        return result

    def process_hit(self, hit_id, assignments):
        processed_results = []

        logger.info('processing image hit %s' % hit_id)
        # get a list of the box ids for this hit from the first assignment in
        # the assignments
        image_ids = assignments[0]['image_ids'][0].split('|')
        # combine all the assignments' results into a single list
        all_clicked_images = reduce(
            lambda x, y: list(set(x) | set(y.keys())), assignments)
        # This assumes that all the image ids are of the form image_<id>
        # It would result in a bug if we start having  image_ in the <id>
        # portion of the key
        all_clicked_images = [k.replace('image_', '')
                              for k in all_clicked_images
                              if self.image_pat.search(k)]

        # guaranteed 3/3 if nobody clicked
        false_image_ids = set(image_ids) - set(all_clicked_images)
        for image_id in false_image_ids:
            folder_id, thumbnail = self._split_image_id(image_id)
            processed_results.append([hit_id, folder_id, thumbnail,
                                      self.target_label_id, False])
        true_results = defaultdict(int)
        for assignment in assignments:
            for image_id in all_clicked_images:
                munged = "image_%s" % image_id
                if munged in assignment.keys():
                    true_results[image_id] += 1
        for image_id, num_true in true_results.iteritems():
            folder_id, thumbnail = self._split_image_id(image_id)
            num_false = self.max_assignments - num_true
            if num_true >= self.match_threshold:
                result = True
            elif num_false >= self.match_threshold:
                result = False
            else:
                result = None
            processed_results.append([hit_id, folder_id, thumbnail,
                                      self.target_label_id, result])
        return processed_results

    def ingest_on_demand_hit(self, hit_id, assignments):
        return self.process_hit(hit_id, assignments)

    def extract_sub_hits(self, hit_id, assignments):
        sub_hits = defaultdict(list)
        for assignment in assignments:
            time_elapsed, worker_id = assignment['time_elapsed'], \
                assignment['worker_id']
            item_ids = assignment['image_ids'][0].split('|')
            all_clicked_items = [k.replace('image_', '')
                                 for k in assignment
                                 if self.image_pat.search(k)]
            false_item_ids = set(item_ids) - set(all_clicked_items)
            items = [(self.false_answer, false_id)
                     for false_id in false_item_ids]
            items += [(self.true_answer,
                       click_id) for click_id in all_clicked_items]
            time_elapsed = int(ceil(time_elapsed / float(len(item_ids))))
            for response, item_id in items:
                sub_hit_id = hit_id + '_' + item_id
                sub_hits[sub_hit_id].append({
                    'worker_id': worker_id,
                    'time_elapsed': time_elapsed,
                    'answer': response,
                })
        return sub_hits


class WebPageTextEvaluator(MechanicalTurkEvaluator):

    """An MTurk template where we display a screen shot of a web page and ask a yes/no question.
    For example, "Does this web page have family and parenting content?"
    """
    __mapper_args__ = dict(polymorphic_identity='webpage_text')

    template = "webpage_keyword_question_form.html"
    default_keywords = "Categorization,Videos,Tag,Label,Keyword,Image,Screenshot"
    evaluator_type = "page_text"
    hit_type = PageHit
    image_pat = '_'

    def __init__(self, **kwargs):
        super(WebPageTextEvaluator, self).__init__(**kwargs)
        label = self.target_label
        if label is None and self.target_label_id is not None:
            label = Label.get(self.target_label_id)
        if label is not None:
            if self.question is None:
                self.question = 'Does this web page contain %s content?' % label.name
            if self.title is None:
                self.title = 'Web Page Categorization (%s)' % label.name
            if self.description is None:
                self.description = ('You will be shown a screen shot of a web page and asked '
                                    'whether the web page contains %s content' % label.name)
            self.require_adult = True

        if self.keywords is None:
            self.keywords = self.default_keywords

    @staticmethod
    def extract_hit_data(hit):
        return {"page_id": hit.page_id}, hit.label

    @staticmethod
    def process_webpage_title(title):
        """ Replace non-unicode characters by the closest matching unicode characters
            (or leave blank if no match found) and remove unicode control characters.
        """
        title = unidecode(title).decode('utf-8')
        tbl = {i : None for i in xrange(sys.maxunicode)
            if unicodedata.category(unichr(i)) in ('Cf','Cc')}
        return title.translate(tbl)

    def format_data(self, is_on_demand=False, process_title=False, page_id=None,
            image_id=None):
        msg = 'Function must be called either with page_id OR image_id argument'
        assert (page_id or image_id) and not (page_id and image_id), msg

        data = defaultdict(dict)
        title = u''
        if page_id:
            title = WebPage.get(page_id).title
            data['data']['page_id'] = str(page_id)
        elif image_id:
            data['data']['image_id'] = image_id

        if process_title:
            title = self.process_webpage_title(title)
        bucket = self.get_s3_bucket(is_on_demand=is_on_demand)
        data.update({
            'title': title,
            'question': self.question,
            'image_bucket': bucket,
        })
        data['data'].update({
            "clickable": 'false',
            "labels": ["yes", "no"],
            "evaluator_id": str(self.id),
        })
        return data

    def get_result(self, golden_hit_id):
        result = session.query(PageHit.result).join(GoldenHit,
                                                    PageHit.hit_id == GoldenHit.hit_id).filter(
            GoldenHit.golden_hit_id == golden_hit_id).scalar()
        return result

    def process_hit(self, hit_id, assignments):
        processed_results = []
        logger.info('processing page hit %s' % hit_id)
        num_yes = sum((assignment['answer'] ==
                       self.true_answer)
                      for assignment in assignments)
        num_nos = len(assignments) - num_yes
        page_id = int(assignments[0]['page_id'][0])
        if num_yes >= self.match_threshold:
            result = True
        elif num_nos >= self.match_threshold:
            result = False
        else:
            result = None
        processed_results.append((hit_id, page_id, self.target_label_id,
                                  result))
        return processed_results

    def ingest_on_demand_hit(self, hit_id, assignments):
        processed_results = []
        logger.info('processing on-demand page hit %s' % hit_id)
        num_yes = sum((assignment['answer'] ==
                       self.true_answer)
                      for assignment in assignments)
        num_nos = len(assignments) - num_yes
        image_id = assignments[0]['image_id'][0]
        job_id, resource_id = self._split_image_id(image_id)
        if num_yes >= self.match_threshold:
            result = True
        elif num_nos >= self.match_threshold:
            result = False
        else:
            result = None
        processed_results.append((hit_id, job_id, resource_id, self.target_label_id,
                                  result))
        return processed_results


class WorkerSet(object):

    def __init__(self, hit_id, workers=None, vote=0):
        if workers is None:
            self.workers = {}
        else:
            self.workers = workers
        self.hit_id = hit_id
        self.vote = vote

    def __eq__(self, other):
        flag = True
        for worker_id, ans in self.workers.iteritems():
            if other.workers[worker_id] != ans:
                flag = False
        return self.hit_id == other.hit_id and flag

    def __hash__(self):
        return hash(self.hit_id)

    def add_worker(self, worker_id, worker, answer):
        self.workers[worker_id] = (worker, answer)
        self.vote += answer

    def update_workers(self):
        hits_count = len(self.workers)
        # checking for conflict
        if self.vote != 0 and self.vote != hits_count:
            if self.vote > (hits_count / 2.):  # 'no vote' minority
                for worker, answer in self.workers.itervalues():
                    if not answer:
                        worker.num_minority += 1
            elif self.vote < (hits_count / 2.):  # 'yes vote' minority
                for worker, answer in self.workers.itervalues():
                    if answer:
                        worker.num_minority += 1


def get_hits_by_evaluator():
    """Get results from MTurk and group them as a dict of dicts of answers
    keyed by evaluator_id and then by hit_id.
    For each HIT, there will be one answer per worker.
    The max_assignments setting on the evaluator determines the number of workers.
    """
    failures = []
    results = defaultdict(dict)
    for hit_id, result in MTurkUtils.get_all_reviewable_hits().iteritems():
        try:
            evaluator_id = result[0]['evaluator_id'][0]
        except (IndexError, KeyError, AttributeError):
            logger.exception("Failed to process hit: {}".format(hit_id))
            worker_id = None
            if result and hasattr(result[0], "worker_id"):
                worker_id = result[0]["worker_id"]
            failures.append({"hit_id": hit_id, "worker_id": worker_id})
        else:
            results[evaluator_id][hit_id] = result
    session.bulk_insert_mappings(MTurkHitFailure, failures)
    return results


def flush_completed_hits_to_db():
    """Pull results from MTurk and write them to the database"""
    # list of hit_ids to delete after successful processing
    # n.b. "compdict" is the "completed_dict" structure from MTurkUtils, a dictionary
    # of HIT_id: {answers} where answers is qid: fields
    failures = []
    logger.info('Flushing completed hits to DB')
    for evaluator_id, compdict in get_hits_by_evaluator().iteritems():
        evaluator = MechanicalTurkEvaluator.get(evaluator_id)
        assert evaluator, "No evaluator found for id: {}".format(evaluator_id)
        for hit_id, assignments in compdict.iteritems():
            try:
                evaluator.ingest_hit(hit_id, assignments)
            except (IndexError, KeyError, AttributeError):
                logger.exception("Failed to process hit: {}".format(hit_id))
                worker_id = None
                if assignments and hasattr(assignments[0], "worker_id"):
                    worker_id = assignments[0]["worker_id"]
                failures.append({"hit_id": hit_id, "worker_id": worker_id})
    session.bulk_insert_mappings(MTurkHitFailure, failures)
    logger.info('Finished flushing, updating DB')


HIT_TYPE_TO_EVALUATOR_TYPE = {BoxHit: ClickableBoxEvaluator,
                              VideoHit: VideoCollageEvaluator,
                              ImageHit: ClickableImageEvaluator,
                              PageHit: WebPageTextEvaluator}
