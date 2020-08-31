from datetime import datetime

from affine.model.base import *
from affine.model._sqla_imports import *
from affine.model.secondary_tables import bundle_keywords_table

__all__ = ['UserKeyword', 'KeywordBundle', 'WebPageUserKeywordResult']


class UserKeyword(Base):
    __tablename__ = 'user_keywords'
    id = Column(Integer, nullable=False, autoincrement=True, primary_key=True)
    text = Column(
        VARCHAR(200, charset='utf8', convert_unicode=True, collation="utf8_bin"),
        unique=True, nullable=False
    )
    title_only = Column(Boolean, nullable=False, default=False)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    @classmethod
    def get_or_create(cls, text, title_only=False):
        keyword = cls.query.filter_by(text=text, title_only=title_only).first()
        if not keyword:
            keyword = cls(text=text, title_only=title_only)
            session.flush()
        return keyword

    @classmethod
    def delete(cls, user_keyword):
        cls.query.filter_by(id = user_keyword.id).delete()


class KeywordBundle(Base):
    __tablename__ = 'keyword_bundles'
    id = Column(Integer, nullable=False, autoincrement=True, primary_key=True)
    name = Column(Unicode(255), nullable=False)
    timestamp = Column(Timestamp, nullable=False, server_default=func.now())

    keywords_in_bundle = relation(UserKeyword, secondary = bundle_keywords_table, backref='bundles')


class WebPageUserKeywordResult(Base):
    """A entry signifying that on a specific web page there is a specific user_keyword"""
    __tablename__ = "webpage_user_keyword_results"
    page_id = Column(Integer, ForeignKey('web_pages.id'), nullable = False, primary_key=True)
    user_keyword_id = Column(Integer, ForeignKey('user_keywords.id'), nullable = False, primary_key=True)

    page = relation('WebPage', backref=backref('user_keyword_results', passive_deletes=True))
    user_keyword = relation('UserKeyword', backref=backref('page_results', passive_deletes=True))

    def __unicode__(self):
        url, user_keyword_name = self.page_id, self.user_keyword_id
        if self.page:
            url = "'%s'" % (self.page.remote_id[:100],)
        if self.user_keyword:
            user_keyword_name = self.user_keyword.text
        return u'<user keyword on page (%s, %s)>' % (url, user_keyword_name)

    @classmethod
    def get_result(cls, page_id, user_keyword_id):
        wpukr = cls.query.filter_by(page_id = page_id,
                                    user_keyword_id = user_keyword_id).first()
        return wpukr is not None

    @classmethod
    def set_result(cls, page_id, user_keyword_id):
        cols = 'page_id, user_keyword_id'
        execute("""
            insert ignore into %s (%s)
                values (%s, %s)""" %
            (cls.__tablename__, cols, page_id, user_keyword_id))

    @classmethod
    def clear_result(cls, page_id, user_keyword_id):
        cls.query.filter_by(page_id = page_id, user_keyword_id = user_keyword_id).delete()

    @classmethod
    def load_from_file(cls, wpukr_file, on_duplicate='ignore'):
        """Expects a file with tab separated fields matching the schema of WebPageUserKeywordResults"""
        cols = 'page_id, user_keyword_id'
        cls._load_from_file(wpukr_file, cols, on_duplicate)
