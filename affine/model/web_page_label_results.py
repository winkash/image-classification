from affine.model.base import *
from affine.model._sqla_imports import *

__all__ = ['WebPageLabelResult']


class WebPageLabelResult(Base):
    """A boolean result of a specific web page for a specific label"""
    __tablename__ = "webpage_label_results"
    page_id = Column(Integer, ForeignKey('web_pages.id'), nullable = False, primary_key=True)
    label_id = Column(Integer, ForeignKey('labels.id'), nullable = False, primary_key=True)

    page = relation('WebPage', backref=backref('label_results', passive_deletes=True))
    label = relation('Label', backref=backref('page_results', passive_deletes=True))

    def __unicode__(self):
        url, label_name = self.page_id, self.label_id
        if self.page:
            url = "'%s'" % (self.page.remote_id[:100],)
        if self.label:
            label_name = self.label.name
        return u'<page result (%s, %s)>' % (url, label_name)

    @classmethod
    def get_result(cls, page_id, label_id):
        wplr = cls.query.filter_by(page_id = page_id,
                                   label_id = label_id).first()
        return wplr is not None

    @classmethod
    def set_result(cls, page_id, label_id):
        kwargs = {'page_id' : page_id, 'label_id' : label_id}
        if cls.query.filter_by(**kwargs).first() is None:
            cls(**kwargs)
        session.flush()

    @classmethod
    def load_from_file(cls, wplr_file, on_duplicate='ignore'):
        """Expects a file with tab separated fields matching the schema of WebPageLabelResults"""
        cols = 'page_id, label_id'
        cls._load_from_file(wplr_file, cols, on_duplicate)
