from logging import getLogger

import json
import os
import random
import tarfile

from affine import config
from affine.aws import s3client
from affine.retries import retry_operation
from apiclient.discovery import build
from apiclient.errors import HttpError

__all__ = ['YoutubeCollector', 'YoutubeVideoText', 'create_traintest_json']

logger = getLogger(__name__)


def create_traintest_json(config_dict):
    yc = YoutubeCollector()
    logger.info("downloading negative examples from S3")
    yc.write_neg_json(config_dict)
    logger.info("done downloading negative examples from S3")
    logger.info("retrieving positive examples from Youtube")
    yc.write_pos_json(config_dict)
    logger.info("done retrieving positive examples from Youtube")


class YoutubeCollector(object):

    YT_MAX_LIMIT = 950

    def __init__(self):
        DEVELOPER_KEY = config.get('youtube.developer_key')
        YOUTUBE_API_SERVICE_NAME = "youtube"
        YOUTUBE_API_VERSION = "v3"
        self.yt_service = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                                developerKey=DEVELOPER_KEY)

    def get_youtube_video_ids(self, query_string, n_results):
        """Returns a list of youtube video ids for training"""
        max_results = 50
        page_token = None
        video_ids = []
        # might not be a limitation in youtube API v3
        n_results = min(n_results, self.YT_MAX_LIMIT)
        video_ids = []
        for i in range(0, n_results + max_results, max_results):
            search_request = self.yt_service.search().list(
                q=query_string,
                part="id",
                maxResults=50,
                type="video",
                pageToken=page_token,
            )
            search_response = retry_operation(search_request.execute, sleep_time=0.5, error_class=HttpError)
            video_ids.extend([res['id']['videoId'] for res in search_response.get('items', [])])
            page_token = search_response['nextPageToken']
        video_ids = list(set(video_ids))  # may not be needed?
        return video_ids[:n_results]

    def write_pos_json(self, config_dict):
        """Creates train/test files containing positive data"""
        query_dict = config_dict['query_dict']
        test_split = config_dict['test_split']
        pos_train_json = config_dict['pos_train_json']
        pos_test_json = config_dict['pos_test_json']
        fo_train = open(pos_train_json, 'w')
        fo_test = open(pos_test_json, 'w')
        for q in query_dict:
            video_ids = self.get_youtube_video_ids(q, query_dict[q])
            random.shuffle(video_ids)
            train_size = int(len(video_ids)*(1.0-test_split))
            for v_id in video_ids[:train_size]:
                try:
                    yvt = self.build_youtube_video_text(v_id, include_related=config_dict['include_related'])
                    fo_train.write(yvt.to_json()+'\n')
                except Exception, err:
                    logger.error(err)

            for v_id in video_ids[train_size:]:
                try:
                    yvt = self.build_youtube_video_text(v_id, include_related=config_dict['include_related'])
                    fo_test.write(yvt.to_json()+'\n')
                except Exception, err:
                    logger.error(err)
        fo_train.close()
        fo_test.close()

    def write_neg_json(self, config_dict):
        """Creates train/test files containing negative data"""
        skip_labels = [int(i) for i in config_dict['youtube_data_ignore_labels']]
        bucket = config.s3_detector_bucket()
        tarball_name = config_dict['neg_tarball_s3']
        fname = tarball_name.rstrip('.tar.gz')
        s3client.download_from_s3(bucket, tarball_name, tarball_name)
        with tarfile.open(tarball_name, 'r:*') as tar:
            tar.extractall()
        assert os.path.isfile(fname)

        test_split = config_dict['test_split']
        neg_train_json = config_dict['neg_train_json']
        neg_test_json = config_dict['neg_test_json']
        fo_train = open(neg_train_json, 'w')
        fo_test = open(neg_test_json, 'w')
        count = 0
        skip_labels_set = set(skip_labels)
        skip_lines = set()
        for lnum, jsn in enumerate(open(fname)):
            yvt = YoutubeVideoText.to_object(jsn)
            if yvt.label_id in skip_labels_set:
                skip_lines.add(lnum)
            count += 1

        rr = range(count)
        random.shuffle(rr)
        line_nums = list(set(rr) - skip_lines)
        test_size = len(line_nums)*test_split
        test_line_nums = set(line_nums[:int(test_size)])
        for lnum, jsn in enumerate(open(fname)):
            if lnum in skip_lines:
                continue
            if lnum in test_line_nums:
                fo_test.write(jsn)
            else:
                fo_train.write(jsn)

        fo_train.close()
        fo_test.close()

    def build_youtube_video_text(self, v_id, include_related=False):
        yvt = YoutubeVideoText(v_id)
        search_request = self.yt_service.videos().list(
            part="snippet",
            id=v_id,
        )
        search_response = retry_operation(search_request.execute, sleep_time=0.5, error_class=HttpError)
        yvt.video_title = search_response['items'][0]['snippet']['title']
        yvt.video_description = search_response['items'][0]['snippet']['description']
        yvt.video_comments = self.get_comments(v_id)
        if include_related:
            yvt.related_videos_text = self.get_related_videos_text(v_id)
        return yvt

    def get_related_videos_text(self, video_id):
        search_request = self.yt_service.search().list(
            part="snippet",
            maxResults=10,
            type="video",
            relatedToVideoId=video_id,
        )
        search_response = retry_operation(search_request.execute, sleep_time=0.5, error_class=HttpError)
        related_videos_text = []
        for res in search_response.get('items', []):
            related_videos_text.append('%s %s'%(res['snippet']['title'], res['snippet']['description']))
        return related_videos_text

    def get_comments(self, video_id):
        comments = []
        search_request = self.yt_service.commentThreads().list(
            part="snippet",
            maxResults=10,
            videoId=video_id,
            textFormat="plainText"
        )
        search_response = retry_operation(search_request.execute, sleep_time=0.5, error_class=HttpError)
        for res in search_response.get('items', []):
            comments.append(res['snippet']['topLevelComment']['snippet']['textDisplay'])
        return comments


class YoutubeVideoText(object):
    '''Class for the text attributes of a Youtube video
    '''

    def __init__(self, video_id):
        self.video_id = video_id
        self.video_title = ''
        self.video_description = ''
        self.video_comments = []
        self.related_videos_text = []
        self.label_id = None

    def __unicode__(self):
        return '<YoutubeVideoText(%s)>' %self.video_id

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def to_object(cls, json_str):
        return json.loads(json_str, object_hook=cls._json_object_hook)

    @classmethod
    def _json_object_hook(cls, d):
        inst = cls(video_id=d.pop('video_id'))
        for k in d:
            setattr(inst, k, d[k])
        return inst
