#!/usr/bin/env python
import time
import os

from affine.model import session, Label, MTurkImageJob, ImageHit, VideoHit, \
    Video, WebPage, VideoOnPage
from script import image_annotation_tier1
from affine.model.mturk.evaluators import HIT_TYPE_TO_EVALUATOR_TYPE

SLEEP_TIME = 300
EVERY_N_SEC = 5

__all__ = ['get_images', 'get_vids', 'mturk_submission_ingestion',
           'mturk_submission_only', 'ingest_all_results']


def get_images(all_urls):
    ''' get a set of images that will be submitted to mturk '''
    image_set = []
    if all_urls:
        video_ids = get_vids(all_urls)
        image_set = image_annotation_tier1.get_images_to_qa(
            video_ids, EVERY_N_SEC)

    return image_set


def get_vids(all_urls):
    """ get a list of video ids that are shown in any of the given urls """
    video_ids = []
    if all_urls:
        query = session.query(Video.id.distinct()).join(
            VideoOnPage).filter_by(active=True, is_preroll=False)
        query = query.join(WebPage).filter(WebPage.remote_id.in_(all_urls))
        video_ids = [v for (v,) in query]

    return video_ids


def get_wp_ids(all_urls):
    """ get a list of webpage ids that correspond to each of the given urls """
    wp_ids = []
    if all_urls:
        query = session.query(WebPage.id).\
            filter(WebPage.remote_id.in_(all_urls))
        wp_ids = [wp for (wp,) in query]

    return wp_ids


def ingest_all_results(job):
    """ 
    Waits until 95% of the ImageHit results have been ingested, 
    if there are any ImageHits for the input job
    """
    while True:
        MTurkImageJob.ingest_results()
        complete_hits = ImageHit.query.filter_by(
            mturk_image_job_id=job.id, outstanding=False).count()
        total_hits = ImageHit.query.filter_by(
            mturk_image_job_id=job.id).count()
        job.update_status()

        if total_hits > 0:
            comp_hits = float(complete_hits) / float(total_hits)
        else:
            comp_hits = 1

        print "%i%% of hits responded" % (comp_hits * 100)
        if comp_hits < 0.95:
            time.sleep(SLEEP_TIME)
        else:
            break


def mturk_submission_ingestion(super_config, inter_dir):
    """
    Read from the DB the video frames that correspond to the injected urls 
    submit image hits to mturk, and ingest the results

    NOTE: This script only submits (and ingests) ImageHits. 
    It's left for back-compatibility purposes. We dont need to manually ingest 
    the expected results because cron ingestion job runs automatically every few hours
    """

    url_file = os.path.join(inter_dir, 'all_urls.txt')
    all_urls = []
    with open(url_file, 'r') as fo:
        all_urls = [x.strip() for x in fo.readlines()]

    job, hit_type, num_hits_submitted = mturk_submission_only(
        super_config, all_urls, hit_type=ImageHit)
    if job:
        ingest_all_results(job)
        return job.id
    else:
        return -1


def mturk_submission_only(super_config, all_urls, hit_type=ImageHit):
    """
    Read from the video info that corresponds to the injected urls 
    and submit the required hits to mturk

    Args:
        super_config: config object with submission params. 
            It is expected to at least contain:
            super_config['model_to_db_injection']['target_label']
            super_config['mturk_submission_params']['mturk_question']
        all_urls: list of strings with urls
        hit_type: Type of MTurk hit (Only ImageHit and VideoHit are supported)

    Returns:
        job: MTurkImageJob object created
        hit_type: type of submitted hits
        num_hits_submitted: int, number of submitted hits

    """
    assert (hit_type == ImageHit or hit_type == VideoHit), \
        "Only ImageHit or VideoHit are valid types for hit_type param"

    label_name = super_config['model_to_db_injection']['target_label']
    question = super_config['mturk_submission_params']['mturk_question']

    evaluator_type = HIT_TYPE_TO_EVALUATOR_TYPE[hit_type]
    if hit_type == ImageHit:
        hit_data = get_images(all_urls)
    else:
        hit_data = get_vids(all_urls)

    job = None
    if len(hit_data) > 0:
        label = Label.get_or_create(label_name)
        job = MTurkImageJob(label.id, question=question,
                            evaluator_type=evaluator_type)
        hit_type, num_hits_submitted = job.submit_hits(hit_data)

    return job, hit_type, num_hits_submitted
