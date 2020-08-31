"""Web pages that our VCR has visited. They may have videos."""
import csv
import tempfile

from collections import defaultdict
from hashlib import sha1
from datetime import datetime

from sqlalchemy.ext.hybrid import hybrid_property
import sqlalchemy.types as types
import affine.normalize_url as normalize

from sqlalchemy import event
from affine.model.labels import Keyword, WeightedKeyword
from affine.aws import s3client
from affine.model.videos import Video
from affine import config
from affine.model.base import *
from affine.model.load_data_infile import load_data_infile
from affine.model.secondary_tables import *
from affine.model._sqla_imports import *
from affine.retries import retry_operation
from affine.video_processing import resize_image, convert_png_to_jpeg

__all__ = ['WebPage', 'VideoOnPage', 'WebPageInventory', 'DumpFromInfoBright', 'base_query']


class ListOfStrings(types.TypeDecorator):
    impl = types.Unicode

    def process_bind_param(self, value, dialect):
        return u' '.join(value or [])

    def process_result_value(self, value, dialect):
        return (value or '').split()


class WebPage(Base):
    __tablename__ = 'web_pages'
    id = Column(Integer, primary_key=True)
    remote_id = Column(URL, nullable=False)  # normalized URL
    remote_id_sha1 = Column(CHAR(40))
    domain = Column(
        VARCHAR(128, charset='ascii', collation='ascii_bin'), nullable=False)
    s3_screenshot = Column(Boolean, nullable=False, default=False)
    s3_screenshot_full = Column(Boolean, nullable=False, default=False)
    s3_page_source = Column(Boolean, nullable=False, default=False)
    s3_page_text = Column(Boolean, nullable=False, default=False)
    s3_favicon = Column(Boolean, nullable=False, default=False)
    title = Column(UnicodeText, nullable=False, default=u'')
    processed_title = Column(ListOfStrings, default=u'')
    last_crawled_text = Column(DateTime, nullable=False, default=datetime.utcnow())
    last_crawled_video = Column(DateTime, nullable=True, default=None)
    change_count = Column(Integer, nullable=False, default=0)
    crawl_count = Column(Integer, nullable=False, default=0)
    fail_count = Column(Integer, nullable=False, default=0)
    last_label_update = Column(DateTime)
    text_detection_update = Column(DateTime)
    last_detection_at_llu = Column(DateTime)
    last_text_detection_at_llu = Column(DateTime)
    last_score_update = Column(DateTime)

    def __unicode__(self):
        return u'<WebPage(%s)>' % self.remote_id

    @hybrid_property
    def preroll_ok(self):
        '''conditions when page can be used for decisioning as a preroll page'''
        return (self.last_label_update != None) & \
            (self.last_text_detection_at_llu != None) & \
            (self.last_detection_at_llu != None)

    @hybrid_property
    def nonpreroll_ok(self):
        '''conditions when page can be used for decisioning as a non-preroll page'''
        return (self.last_label_update != None) & \
            (self.last_text_detection_at_llu != None) & \
            (self.last_crawled_video != None)

    @property
    def active_videos(self):
        return [crawled.video for crawled in self.crawled_videos if crawled.active and not crawled.is_preroll]

    @classmethod
    def by_url(cls, url, session=session):
        remote_id = normalize.parse_url(url)
        obj = session.query(cls).filter_by(remote_id=remote_id).first()
        return obj

    @property
    def prerolls(self):
        return [crawled.video for crawled in self.crawled_videos if crawled.is_preroll]

    def new_crawl(self, videos, prerolls=None):
        """We have just visited this page and found the given videos and prerolls.
        The videos passed in should be unique (no duplicates).
        """
        update_args = {
            'last_crawled_video': datetime.utcnow(),
            'crawl_count': WebPage.crawl_count + 1,
            'text_detection_update': None,
        }
        if self.crawl_count:
            active_videos = [
                video for (video, stream_url, is_autoplay, width, height, top, left) in videos]
            if set(active_videos) != set(self.active_videos):
                update_args['change_count'] = WebPage.change_count + 1
        query = WebPage.query.filter_by(id=self.id)
        retry_operation(query.update, update_args)

        for crawled_video in self.crawled_videos:
            crawled_video.active = False
        for (is_preroll, video_list) in [(False, videos), (True, prerolls or [])]:
            for video, stream_url, is_autoplay, player_width, player_height, player_top, player_left in video_list:
                crawled_video = None
                for old_crawled_video in self.crawled_videos:
                    if old_crawled_video.video == video:
                        crawled_video = old_crawled_video
                        break
                if crawled_video is None:
                    crawled_video = VideoOnPage(
                        page=self, video=video, seen_count=0)
                crawled_video.active = not is_preroll
                crawled_video.is_preroll = is_preroll
                crawled_video.seen_count += 1
                crawled_video.stream_url = stream_url
                crawled_video.is_autoplay = is_autoplay
                crawled_video.player_width = player_width
                crawled_video.player_height = player_height
                crawled_video.player_left = player_left
                crawled_video.player_top = player_top

        session.flush()

    @classmethod
    def get_or_create(cls, remote_id):
        remote_id = normalize.parse_url(remote_id)
        page = (cls.query.filter_by(remote_id=remote_id).first() or
                cls(remote_id=remote_id))
        session.flush()
        return page

    @property
    def description_text(self):
        return self.get_page_text()

    @property
    def processed_description_text(self):
        text = self.get_page_text()
        from affine.detection.nlp.keywords.keyword_matching import process_text
        if text is not None:
            return process_text(text)
        return None

    @description_text.setter
    def description_text(self, text):
        self.upload_page_text(text)
        self.mark_page_text_uploaded()

    @property
    def title_and_text(self):
        text = self.title or u''
        desc_text = self.description_text
        if desc_text:
            text += ' ' + desc_text
        return text

    def s3_screenshot_url(self):
        if self.s3_screenshot:
            bucket = config.s3_bucket()
            return "http://%s.s3.amazonaws.com/screenshot/%s" % (bucket, self.id)

    def s3_screenshot_full_url(self):
        if self.s3_screenshot_full:
            bucket = config.s3_bucket()
            return "http://%s.s3.amazonaws.com/screenshot_full/%s" % (bucket, self.id)

    def s3_page_source_url(self):
        if self.s3_page_source:
            bucket = config.s3_bucket()
            return "http://%s.s3.amazonaws.com/page_source/%s" % (bucket, self.id)

    def s3_page_text_url(self):
        if self.s3_page_text:
            bucket = config.s3_bucket()
            return "http://%s.s3.amazonaws.com/page_text/%s" % (bucket, self.id)

    def s3_favicon_url(self):
        if self.s3_favicon:
            bucket = config.s3_bucket()
            return "http://%s.s3.amazonaws.com/favicon/%s" % (bucket, self.id)

    def get_page_text(self):
        if self.s3_page_text:
            bucket = config.s3_bucket()
            urlpath = "%s/%s" % ('page_text', self.id)
            text = s3client.download_from_s3_as_string(bucket, urlpath)
            return text.decode('utf-8')
        return None

    def upload_screenshot(self, path):
        bucket = config.s3_bucket()
        urlpath = "%s/%s" % ('screenshot', self.id)
        s3client.upload_to_s3(bucket, urlpath, path, public=True)
        path = resize_image(path)
        s3client.upload_to_s3(bucket, urlpath + '_thumb', path, public=True)

    def upload_screenshot_full(self, path):
        bucket = config.s3_bucket()
        urlpath = "%s/%s" % ('screenshot_full', self.id)
        thumb_path = resize_image(path)
        s3client.upload_to_s3(bucket, urlpath + '_thumb', thumb_path, public=True)
        convert_png_to_jpeg(path, path, quality=60)
        s3client.upload_to_s3(bucket, urlpath, path, public=True)

    def upload_page_source(self, path):
        bucket = config.s3_bucket()
        urlpath = "%s/%s" % ('page_source', self.id)
        s3client.upload_to_s3(bucket, urlpath, path, public=True)

    def upload_page_text(self, text):
        bucket = config.s3_bucket()
        urlpath = "%s/%s" % ('page_text', self.id)
        text = text.encode('utf-8')
        s3client.upload_to_s3_from_string(bucket, urlpath, text, public=True)

    def upload_favicon(self, path):
        bucket = config.s3_bucket()
        urlpath = "%s/%s" % ('favicon', self.id)
        s3client.upload_to_s3(bucket, urlpath, path, public=True)

    def mark_screenshot_uploaded(self):
        self.s3_screenshot = True

    def mark_screenshot_full_uploaded(self):
        self.s3_screenshot_full = True

    def mark_page_source_uploaded(self):
        self.s3_page_source = True

    def mark_page_text_uploaded(self):
        self.s3_page_text = True

    def mark_favicon_uploaded(self):
        self.s3_favicon = True

    @staticmethod
    def update_domain_and_sha1(target, remote_id, old_remote_id, initiator):
        if remote_id is not None:
            target.domain = normalize.domain_of_url(remote_id)
            target.remote_id_sha1 = sha1(remote_id).hexdigest()

    @staticmethod
    def update_processed_title(target, title, old_title, initiator):
        from affine.detection.nlp.keywords.keyword_matching import process_text
        if title is not None:
            target.processed_title = process_text(title)

    def keyword_results(self, label_id):
        """ get list of keywords present on the web page that fired the label"""
        from affine.detection.nlp.keywords.keyword_matching import \
                PageKeywordMatcher
        body_matcher = PageKeywordMatcher()
        title_matcher = PageKeywordMatcher()
        weighted_keywords = session.\
            query(Keyword.text, WeightedKeyword.title_weight,
                  WeightedKeyword.body_weight).\
            join(WeightedKeyword).filter_by(label_id=label_id)
        for kw_text, title_weight, body_weight in weighted_keywords:
            title_matcher.add_keyword((kw_text, title_weight), kw_text)
            body_matcher.add_keyword((kw_text, body_weight), kw_text)
        matches = defaultdict(int)
        for kw_text, title_weight in title_matcher.matching_keywords(
                self.processed_title):
            matches[kw_text] = title_weight
        for kw_text, body_weight in body_matcher.matching_keywords(
                self.processed_description_text):
            if kw_text not in matches or matches[kw_text] == 0:
                matches[kw_text] = body_weight
        return matches.items()


# Cascade and set domain, sha1 when the remote_id is set/ updated
event.listen(WebPage.remote_id, 'set', WebPage.update_domain_and_sha1)
# Cascade and set the processed_title when the title is set/ updated
event.listen(WebPage.title, 'set', WebPage.update_processed_title)


class VideoOnPage(Base):
    __tablename__ = 'video_pages'
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False)
    page_id = Column(Integer, ForeignKey('web_pages.id'), nullable=False)
    active = Column(Boolean, nullable=False, default=True)
    seen_count = Column(Integer, nullable=False, default=1)
    stream_url = Column(VARCHAR(4096))

    is_preroll = Column(Boolean, nullable=False, default=False)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())
    # these are null for rows that existed before we starting tracking this
    is_autoplay = Column(Boolean)
    player_height = Column(Integer)
    player_width = Column(Integer)
    player_top = Column(Integer)
    player_left = Column(Integer)

    video = relation(
        'Video', backref=backref('crawled_pages', cascade='all,delete-orphan'))
    page = relation('WebPage', backref=backref(
        'crawled_videos', cascade='all,delete-orphan'))

    __table_args__ = (
        UniqueConstraint('video_id', 'page_id', name='ix_video_id_page_id'), {})

    @classmethod
    def get_or_create(cls, video_id, page_id):
        # Persist any unflushed changed so they don't get lost if we hit an
        # IntegrityError and rollback
        session.flush()
        try:
            vop = cls(video_id=video_id, page_id=page_id)
            session.flush()
        except IntegrityError:
            return cls.query.filter_by(video_id=video_id, page_id=page_id).first()
        else:
            return vop


class DumpFromInfoBright(object):

    @classmethod
    def _load_data(cls, data):
        """Populate given column values in the table, given as a list of tuples"""
        tablename = cls.__tablename__
        cols = cls.COLUMNS_FOR_DATA_LOAD
        with tempfile.NamedTemporaryFile() as csv_file:
            writer = csv.writer(csv_file, delimiter='\t')
            writer.writerows(data)
            csv_file.flush()
            # num_tries=1 because we don't want to retry since we are wrapped in
            # a larger transaction
            load_data_infile(
                tablename, csv_file.name, cols, 'ignore', line_delimiter="\n", num_tries=1)

    @classmethod
    def after_load_into_table(cls):
        """Post-load hook for child classes to override"""

    @classmethod
    def update(cls, data):
        """Populate the table with the supplied data"""
        with session.begin():
            cls.query.delete()
            cls._load_data(data)
            cls.after_load_into_table()


class WebPageInventory(Base):
    """Summarizes impression count for each page"""
    __tablename__ = "web_page_inventory"
    page_id = Column(Integer, ForeignKey("web_pages.id"), primary_key=True)
    count = Column(Integer, nullable=False)  # Impressions per month
    video_id = Column(Integer, ForeignKey('videos.id'), default=None)

    page = relation('WebPage', backref='inventory')
    video = relation('Video')

    def __unicode__(self):
        return u'%s for page_id %s, video_id %s and count %s' % (
            self.__class__.__name__, self.page_id, self.video_id, self.count
        )


def base_query(join_to_videos=False, join_to_inventory=True):
    query = session.query(WebPage.id.distinct()).filter(
        WebPage.domain != 'set.tv')
    if join_to_inventory:
        # Include all pages that have inventory
        query = query.join(WebPage.inventory)

    if join_to_videos:
        # Consider pages only having an active video
        query = query.join(VideoOnPage)
        query = query.filter_by(active=True, is_preroll=False)
        query = query.join(Video)

    return query


def get_page_text_dict(page_ids, silent=False):
    """
    Retrieves page_text for given page_ids.
    :param page_ids: List of page_ids to retrieve page text for.
    :param silent: If set to true, will not forward any errors for non-existent page_ids.  Defaults to False and returns
            empty text for non-existent page_ids.
    :return: dictionary with mapping page_id -> page_text.
    """
    page_ids = set(page_ids)
    output = {page_id: "" for page_id in page_ids}
    bucket = config.s3_bucket()
    s3_conn = s3client.connect(bucket)

    for page_id in page_ids:
        urlpath = "%s/%s" % ('page_text', page_id)
        try:
            text = s3_conn.get_key(urlpath).get_contents_as_string()
            output[page_id] = text.decode('utf-8')
        except AttributeError as e:
            if not silent:
                raise e

    return output


def get_page_processed_text_dict(page_ids, silent=False):
    """
    Retrieves processed_text for given page_ids.
    :param page_ids: List of page_ids to retrieve processed text for
    :param silent:  If set to true, will not forward any errors for non-existent page_ids.  Defaults to False and returns
            empty text for non-existent page_ids.
    :return: dictionary with mapping page_id -> processed_text.
    """
    from affine.detection.nlp.keywords.keyword_matching import process_text
    output = get_page_text_dict(page_ids, silent=silent)
    for page_id, page_text in output.iteritems():
        if page_text is not None:
            output[page_id] = process_text(page_text)
    return output
