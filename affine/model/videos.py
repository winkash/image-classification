"""Videos and their frames (images)."""

import os
import shutil
import tempfile
from datetime import datetime
from io import BytesIO

import requests
import PIL.Image

from affine.aws import s3client
from affine import config
from affine.video_processing import extract_images_from_video
from affine.model._sqla_imports import *
from affine.model.base import *
from affine.video_processing import resize_image

__all__ = ['Video', 'Image']


class Video(Base):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True)
    added = Column(DateTime, nullable=False, default=datetime.utcnow)
    height = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    length = Column(Integer, nullable=False) # duration in seconds
    checksum = Column(CHAR(32), unique=True, nullable=False)
    last_detection = Column(DateTime)
    # A checksum of a small part of the file, near the start but past
    # the metadata. Used for detecting when we've failed to retrieve a complete stream
    start_checksum = Column(CHAR(32), unique=True)
    bytes = Column(Integer)
    # Optionally identifies the video from some (normalized / calculated)
    # part of the stream URL
    other_identifier = Column(VARCHAR(500), unique = True)
    s3_video = Column(Boolean, nullable = False, default = False)
    s3_transcript = Column(Boolean, nullable = False, default = False)
    s3_images = Column(VARCHAR(1024), nullable=False, default='')

    def s3_timestamps(self):
        return self.split_timestamps(self.s3_images)

    @staticmethod
    def split_timestamps(s3_images):
        return map(int, filter(None, s3_images.split('|')))

    @property
    def bucket(self):
        return config.s3_bucket()

    @property
    def face_boxes(self):
        from affine.model.boxes import Box
        return Box.query.filter(Box.video_id==self.id,
                                Box.box_type=='Face').all()

    @property
    def text_boxes(self):
        return [box for box in self.boxes if box.box_type == 'Text']

    @property
    def logo_boxes(self):
        return [box for box in self.boxes if box.box_type == 'Logo']

    @staticmethod
    def construct_s3_image_url(video_id, timestamp):
        bucket = config.s3_bucket()
        return "http://%s.s3.amazonaws.com/thumbnail/%s/%s" % (bucket, video_id, timestamp)

    def s3_image_url(self, timestamp):
        return Video.construct_s3_image_url(self.id, timestamp=timestamp)

    @property
    def s3_video_urlpath(self):
        return "%s/%s" % ('video', self.id)

    @property
    def s3_video_url(self):
        if self.s3_video:
            return "http://%s.s3.amazonaws.com/%s" % (self.bucket, self.s3_video_urlpath)

    @property
    def s3_transcript_url(self):
        if self.s3_transcript:
            return "http://%s.s3.amazonaws.com/transcript/%s.txt" % (self.bucket, self.id)

    def upload_image(self, frame, path):
        urlpath = "%s/%s/%s" % ('thumbnail', self.id, frame)
        s3client.upload_to_s3(self.bucket, urlpath, path, public=True)
        path = resize_image(path)
        s3client.upload_to_s3(self.bucket, urlpath + '_thumb', path, public=True)

    def upload_video(self, path):
        s3client.upload_to_s3(self.bucket, self.s3_video_urlpath, path, public=True)

    def upload_transcript(self, path):
        urlpath = "%s/%s.txt" % ('transcript', self.id)
        s3client.upload_to_s3(self.bucket, urlpath, path, public=True)

    def download_video(self, path):
        s3client.download_from_s3(self.bucket, self.s3_video_urlpath, path)

    def download_image(self, timestamp, path):
        urlpath = "%s/%s/%s" %('thumbnail',self.id, timestamp)
        s3client.download_from_s3(self.bucket, urlpath, path)

    def download_and_sample(self):
        fd, video_path = tempfile.mkstemp()
        os.close(fd)
        imagedir = tempfile.mkdtemp()
        try:
            self.download_video(video_path)
            extract_images_from_video(video_path, imagedir)
        except Exception:
            if os.path.exists(video_path):
                os.unlink(video_path)
            if os.path.isdir(imagedir):
                shutil.rmtree(imagedir)
            raise
        return video_path, imagedir

    def mark_images_uploaded(self, frames):
        images = "|".join(map(str, sorted(map(int, set(frames)))))
        self.s3_images = images

    def mark_video_uploaded(self):
        self.s3_video = True

    def mark_transcript_uploaded(self):
        self.s3_transcript = True

    @staticmethod
    def pil_image(video_id, timestamp):
        return Image(video_id, timestamp).pil_image()

    @staticmethod
    def youtube_info_for_video(video_id):
        from affine.model import WebPage
        query = session.query(WebPage.remote_id, WebPage.title).join('crawled_videos').filter_by(video_id = video_id)
        query = query.filter(WebPage.remote_id.like('http://www.youtube.com/watch%'))
        remote_id, title = query.first()
        youtube_id = remote_id[31:42]
        return youtube_id, title

    def youtube_info(self):
        return Video.youtube_info_for_video(self.id)


class Image(object):
    def __init__(self, video_id, time):
        self.video_id = video_id
        self.time = time

    def pil_image(self):
        url = Video.construct_s3_image_url(self.video_id, self.time)
        response = requests.get(url)
        assert response.status_code == 200, 'Got HTTP status code %s' % response.status_code
        data = response.content
        return PIL.Image.open(BytesIO(data))

    def show(self):
        self.pil_image().show()
