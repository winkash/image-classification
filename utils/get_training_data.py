import numpy as np
import tempfile
import os

from datetime import datetime
from logging import getLogger
from sqlalchemy.sql import and_

from ..vision.utils.scene_functions import sample_frames
from affine.model import (Label, ImageHit,
                          VideoHit, WebPageLabelResult, VideoOnPage,
                          LabelHash, MTurkImage, WebPage, session,
                          MTurkImageDetectorResult,
                          ClassifierTarget, AbstractDetector)
from affine.detection.utils import run_mturk_submission, run_url_injection
from affine.detection.vision.utils.scene_functions \
    import get_config

logger = getLogger(__name__)
EXTRA_RATIO = 10

CFG_SPEC = """
    [url_injection]
        query_string = string(min=0, max=400)
        playlist_ids = string(min=0, max=400)
        max_vids = integer(min=1, max=5000, default=100)
        min_num_vids_downloaded = integer(min=1, max=5000, default=30)
        url_injection_priority = integer(min=1, max=10000, default=30)
        time_out = integer(min=6, max=24, default=24)
        overwrite = boolean(default=False)
    [mturk_submission_params]
        mturk_question = string(max=2000)
        hit_type = option(VideoHit, ImageHit)
    [model_to_db_injection]
        target_label = string(min=3, max=50)
    """


def get_negative_examples(label_id, max_videos, n_images, sampling_rate):
    """Get images which are not of a specified label.

    Args:
        label_id: Id of label.
        max_videos: A limit on the number of videos to query.
        n_images: Number of images per video.
        sampling_rate: How often to sample each video, in seconds.

    Returns:
        List of (video id, timestamp)s.
    """
    vdr = session.query(VideoOnPage.video_id.distinct()).filter_by(
        active=True, is_preroll=False)
    vdr = vdr.join(VideoOnPage.page)
    # There must not be a WPLR for the label
    vdr = vdr.outerjoin(WebPageLabelResult, and_(
        WebPageLabelResult.page_id == WebPage.id,
        WebPageLabelResult.label_id == label_id,
    ))
    vdr = vdr.filter(WebPageLabelResult.label_id == None)
    # The page must to be up-to-date for the label
    # This means it is really "false" for the label rather than
    # it being a new page where we don't know yet.
    # Note that we are comparing to the latest timestamps (not the query
    # service timestamps) on the label hash, so this may not find negative
    # examples if the label or detectors changed recently.
    label_hash = LabelHash.get(label_id)
    if label_hash is None:
        vdr = vdr.filter(WebPage.last_label_update != None)
    else:
        vdr = vdr.filter(
            WebPage.last_label_update > label_hash.hash_updated_at)
        if label_hash.latest_detector is not None:
            vdr = vdr.filter(
                WebPage.last_detection_at_llu > label_hash.latest_detector)
        if label_hash.latest_text_detector is not None:
            vdr = vdr.filter(
                WebPage.last_text_detection_at_llu > label_hash.latest_text_detector)
    vdr = vdr.order_by(VideoOnPage.video_id.desc()).limit(max_videos)
    video_ids = [v for (v,) in vdr]
    return sample_frames(video_ids, n_images, sampling_rate)


def get_image_detector_mturk_images(det_id, result):
    """ Get QAed images for an image detector.

    Args:
        det_id: Detector id.
        result: Result from mturk (either None, True, or False).
                E.G. result = True gives true positives and
                result = None gives conflicts.

    Returns: [(video_id, timestamp), ...]
    """
    clf_targets = AbstractDetector.get(det_id).clf_targets
    assert len(clf_targets) == 1, \
        "cannot use method if detector has multiple targets"
    clf_target = clf_targets[0]

    midr = MTurkImageDetectorResult
    mi = MTurkImage
    query = session.query(mi.video_id, mi.timestamp)
    query = query.join(midr, midr.mturk_image_id == mi.id)
    query = query.filter(midr.clf_target_id == clf_target.id).filter(
        mi.result == result)
    query = query.join(ImageHit).filter(ImageHit.outstanding == False)
    return query.all()


def get_list_of_videoids(
    target_label_id=None, target_result=True, excluded_label_list=[],
    start_date=datetime(2013, 1, 1), maxNumVideos=100
):
    """
    Returns a list of video ids that correspond to MTurk Hits done for target_label_id.
    If target_result is True they correspond to True Positives; if False, correspond to False Positives.

    If target_label_id is None, then we get True positives for any random label (except those indicated in the excluded label list)

    Note: maxNumVideos is an upper bound, we may get less video ids, either because they do not exist
        or because there were a lot of duplicated ids in the query result

    Args:

        target_label_id: label id that we want to get.
            IF it's == -1, we'll get random videos EXCEPT those with label id 'negative_label'

        maxNumVideos: maximum number of video ids that we want to get

        excluded_label_list:
            if label_id == None, this function will return videos with any
                label id except the label ids in this excluded_label_list
            (e.g., if we want negative training for soccer,
                excluded_label_list will have soccer and sport labels ids)
        start_date: datetime object.
           date that specifies the start date of the period when mturk hits where obtained

        target_result: target result we want to get from MTurk
            (either True: True Positives, or False: False Positives)
            By default we'll get True Positives

    Returns:
        list of video ids matching the input parameters requirements

    Raises/Assertions:

    """
    assert (type(start_date) == datetime and start_date <= datetime.today()), \
        "start_date should be a datetime object and can't be in the future"

    assert (not target_label_id or Label.get(target_label_id)), \
        "target_label_id should correspond to a Label id in the DB or be None"

    if target_label_id:
        logger.info("obtaining %d '%s positive' video ids for label %s" %
                    (maxNumVideos, str(target_result),
                        Label.get(target_label_id).name))

        queryVideos = VideoHit.query.\
            filter(VideoHit.result == target_result,
                   VideoHit.label_id == target_label_id,
                   VideoHit.timestamp >= start_date).\
            limit(maxNumVideos * EXTRA_RATIO)
    else:
        logger.info(
            'obtaining %d recent video ids from any label except label id in %s' %
            (maxNumVideos, str(excluded_label_list)))

        queryVideos = VideoHit.query.\
                filter(VideoHit.result == True,
                       VideoHit.timestamp >= start_date)

        if len(excluded_label_list) > 0:
            queryVideos = queryVideos.filter(~VideoHit.label_id.in_(excluded_label_list))
        else:
            queryVideos = queryVideos.limit(maxNumVideos * EXTRA_RATIO)

    video_ids = np.unique([v.video_id for v in queryVideos])
    video_ids = list(video_ids[0:maxNumVideos])

    return video_ids


def get_training_data_from_youtube(config_file):
    """
    Run a small pipeline to query, ingest and QA videos from Youtube from a query specified in a config file.
    Args:
        config_file: path to config file including target label info, query details
            (keywords, playlist, hit type, ...).
            For more details, see the GetTrainingConfigfile example.
    """
    config = get_config(config_file, CFG_SPEC.split('\n'))
    hit_type = config['mturk_submission_params']['hit_type']
    if hit_type == 'VideoHit':
        hit_type = VideoHit
    else:
        hit_type = ImageHit

    all_files_dir = tempfile.mkdtemp()
    logger.info(
        " Query Youtube for Urls and inject them to the download queue.")
    injected_urls = run_url_injection.query_video_urls(config, all_files_dir)
    url_file = os.path.join(all_files_dir, 'all_urls.txt')
    all_urls = []
    with open(url_file, 'r') as fo:
        all_urls = [x.strip() for x in fo.readlines()]

    logger.info(
        "Waiting until a reasonable amount of urls are already ingested.")
    d_start = datetime.now()
    num_vids = run_url_injection.check_videos_downloaded(
        all_urls, config, d_start)

    logger.info("Most part (%d) of the videos are downloaded." % (num_vids))
    job, _, num_hits_submitted = \
        run_mturk_submission.mturk_submission_only(config, all_urls, hit_type)
    logger.info("Submitted %d %s" % (num_hits_submitted, str(hit_type)))


    return job, injected_urls, all_urls
