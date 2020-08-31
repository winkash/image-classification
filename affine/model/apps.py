"""Apps meta info"""
from affine.aws import s3client
from affine import config
from affine.model.base import *
from affine.model._sqla_imports import *
from affine.video_processing import resize_image

__all__ = ['App']

class App(Base):
    __tablename__ = 'apps'
    id = Column(Integer, primary_key=True)
    name = Column(URL, nullable=False, unique=True)
    s3_screenshot = Column(Boolean, nullable=False, default=False)
    display_name = Column(UnicodeText, nullable=True)

    def __unicode__(self):
        return u'<App(%s)>' % self.name

    def s3_screenshot_url(self, for_new_screenshot=False):
        if self.s3_screenshot or for_new_screenshot:
            bucket = config.s3_bucket()
            return "http://%s.s3.amazonaws.com/screenshot/app/%s" % (bucket, self.id)

    def upload_screenshot(self, path):
        bucket = config.s3_bucket()
        urlpath = "%s/%s" % ('screenshot/app', self.id)
        s3client.upload_to_s3(bucket, urlpath, path, public=True)
        path = resize_image(path)
        s3client.upload_to_s3(bucket, urlpath + '_thumb', path, public=True)

    def mark_screenshot_uploaded(self):
        self.s3_screenshot = True
