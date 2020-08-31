from sqlalchemy import Integer, Column, BigInteger
from sqlalchemy.dialects.mysql import ENUM

from affine.model.globals import countries
from affine.model.base import Base
from affine.model import execute
from affine.model.web_pages import DumpFromInfoBright

class Forecasting(Base):
    __tablename__ = "forecasting"

    """Summarizes impression count for each page"""
    id = Column(Integer, primary_key=True, autoincrement=True)
    page_id = Column(Integer, nullable=True, primary_key=True, default=None)
    count = Column(BigInteger, nullable=False)  # Impressions per month
    video_id = Column(Integer, nullable=True, default=None, primary_key=True)
    app_id = Column(Integer, nullable=True, primary_key=True, default=None)
    country = Column(ENUM(*countries), nullable=False, primary_key=True)



class ForecastingTemp(DumpFromInfoBright, Base):
    __tablename__ = "forecasting_temp"

    """Summarizes impression count for each page"""
    id = Column(Integer, primary_key=True, autoincrement=True)
    page_id = Column(Integer, primary_key=True, nullable=True, default=None)
    count = Column(BigInteger, nullable=False)  # Impressions per month
    video_id = Column(Integer, nullable=True, primary_key=True, default=None)
    app_id = Column(Integer, nullable=True, primary_key=True, default=None)
    country = Column(ENUM(*countries), nullable=False, primary_key=True)

    COLUMNS_FOR_DATA_LOAD = 'country, page_id, app_id, count'


    @classmethod
    def _get_video_ids(cls):
        tablename = cls.__tablename__
        videotemp = '%s_video_ids' % cls.__tablename__

        execute("TRUNCATE TABLE " + videotemp)
        execute("""
            INSERT INTO {0} (page_id, video_id)
            SELECT video_pages.page_id, MAX(video_pages.video_id) AS video_id
            FROM {1} JOIN video_pages
            ON {1}.page_id = video_pages.page_id
            WHERE video_pages.active = 1 AND video_pages.is_preroll = 0
            GROUP BY video_pages.page_id
        """.format(videotemp, tablename))

    @classmethod
    def update(cls, *args, **kw):
        # _get_video_ids before calling parent's update() method because
        # this fills the temp table that only we touch, so it doesn't need
        # to be inside the transaction
        cls._get_video_ids()
        return super(ForecastingTemp, cls).update(*args, **kw)

    @classmethod
    def _set_video_ids(cls):
        """Set the values in the video_id column based on video_pages"""
        tablename = cls.__tablename__
        temptable = '%s_video_ids' % tablename
        execute("""
            UPDATE %(tablename)s, %(temptable)s
            SET %(tablename)s.video_id = %(temptable)s.video_id
            WHERE %(tablename)s.page_id = %(temptable)s.page_id
        """ % {
            'tablename': tablename,
            'temptable': temptable,
        })

    @classmethod
    def after_load_into_table(cls):
        cls._set_video_ids()
        cls._rename_to_most_updated()

    @classmethod
    def _rename_to_most_updated(cls):
        tablename = cls.__tablename__
        rename_table = tablename.replace("_temp", "")
        execute("RENAME TABLE `{rename_table}` TO `forecasting_tmp_table_33dc3dwv`, `{new_table}` TO `{rename_table}`, `forecasting_tmp_table_33dc3dwv` TO {new_table}".format(new_table=tablename, rename_table=rename_table))
