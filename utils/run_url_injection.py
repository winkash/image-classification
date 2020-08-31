from datetime import datetime
import time
import os
from logging import getLogger

from configobj import ConfigObj

import affine.normalize_url as normalize
from affine.external.crawl.video_crawler_by_keyword import VideoUrls
from affine.model import session, WebPage
from affine.vcr.constants import DOWNLOAD_STAGES
from affine.vcr.dynamodb import DynamoIngestionStatusClient
from affine.aws import sqs
from affine import config as affine_config


__all__ = ['query_video_urls', 'check_videos_downloaded']

logger = getLogger(__name__)

SLEEP_TIME = 250


def check_videos_downloaded(urls, config_filename, d_start):
    """
    Check that the videos have been downloaded to s3

    Args:
        urls: list of strings with injected urls
        config_filename: config file or config object
        d_start: time right before calling to this method

    Returns:
        int, number of videos already ingested in the DB
    """
    config = ConfigObj(config_filename)
    timeout = int(config["url_injection"]["time_out"])
    MIN_NUM_VIDS_DOWNLOADED = int(
        config["url_injection"]["min_num_vids_downloaded"])

    with open(urls, 'r') as fo:
        myurls = fo.readlines()

    myurls2 = [url.strip() for url in myurls]
    donevid = 0
    ingested = []

    dynamo = DynamoIngestionStatusClient()

    while donevid < MIN_NUM_VIDS_DOWNLOADED:
        donevid = 0
        dynamo_urls = dynamo.batch_get(myurls2)
        downloaded = 0
        for key, value in dynamo_urls.iteritems():
            if value is not None and value['status'] == 'Complete':
                downloaded += 1
        logger.info("downloaded %i urls" % downloaded)
        d_now = datetime.utcnow()

        delta = d_now - d_start
        hours = delta.total_seconds() / 3600.0

        if hours >= timeout or downloaded >= MIN_NUM_VIDS_DOWNLOADED:
            for u in myurls2:
                q = dynamo.get(u)
                if q is not None:
                    if q['status'] == 'Complete':
                        donevid += 1
                        if u not in ingested:
                            ingested.append(u)

            if donevid < MIN_NUM_VIDS_DOWNLOADED and hours >= timeout:
                break
        else:
            time.sleep(SLEEP_TIME)

    with open(urls, 'w') as fo:
        for u in ingested:
            u = normalize.parse_url(u)
            fo.write("%s \n" % u)

    return donevid


def query_video_urls(config, inter_dir):
    """ 
    Get youtube urls, inject into download queue and save .txt with urls
    Args:
        config: config object with the query parameters
        inter_dir:  folder path to store a file with all urls (all_urls.txt)
        overwrite: bool, overwrite the info about a url if it is already in the WebPage table
    Returns:
        injected: number of injected urls
        all_urls: list of urls returned by the queries.
    """
    query_string = config['url_injection']['query_string']
    max_vids = int(config['url_injection']['max_vids'])
    priority = int(config['url_injection']['url_injection_priority'])
    playlists_string = None
    overwrite_old_webpage = True
    if 'playlist_ids' in config['url_injection'].keys():
        playlists_string = config['url_injection']['playlist_ids']
    if 'overwrite' in config['url_injection'].keys():
        overwrite_old_webpage = config['url_injection']['overwrite']

    num_youtube_urls = max_vids
    urls = []
    queries = query_string.split(',')
    if playlists_string:
        playlists = playlists_string.split(',')
    else:
        playlists = [None]
    num_video_query = num_youtube_urls / (len(queries) * len(playlists))
    for pl_id in playlists:
        for q in queries:
            query_urls = VideoUrls.get_youtube_urls(
                kw=q.strip(), num=num_video_query, plist_id=pl_id)
            urls.extend(query_urls)
    all_urls = list(set(urls))

    dynamo = DynamoIngestionStatusClient()
    download_queue = sqs.get_queue(affine_config.sqs_download_queue_name())
    outfile_name = os.path.join(inter_dir, 'all_urls.txt')
    f = open(outfile_name, 'w')
    injected = 0
    urls_to_enqueue = []
    for url in all_urls:
        if overwrite_old_webpage or not WebPage.by_url(url):
            injected += 1
            if dynamo.get(url) is None:
                # Enqueue if we haven't seen the URL yet
                item = {'url': url,
                        'status': 'Queued',
                        'download_stage': DOWNLOAD_STAGES['Text']
                }
                urls_to_enqueue.append(item)
        f.write(url + '\n')
    f.close()
    # Add to Dynamo & SQS
    dynamo.batch_put(urls_to_enqueue)
    for item in urls_to_enqueue:
        sqs.write_to_queue(download_queue, item)
    logger.info('Injected %d urls into the download queue' % (injected))
    logger.info(' %d potential positive video urls collected' %
                (len(all_urls)))

    if "report" not in config:
        config["report"] = {}
    config["report"]["num_potential_pos_vid_urls_collected"] = len(all_urls)

    return injected
