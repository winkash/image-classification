import re
from urlparse import urlparse
from hashlib import sha1

from affine.model.base import session, Base
from affine.model._sqla_imports import *
from affine.normalize_url import parse_url, domain_of_url

__all__ = ['DomainGamesList', 'DomainBlacklist', 'DomainTestSiteList', 'dont_process_url',
           'RotatingContentPage', 'UserInitiatedDomain']


class BlacklistBase(object):
    id = Column(Integer, primary_key=True)
    domain = Column(VARCHAR(4096), nullable=False)
    
    def __unicode__(self):
        return u'<%s: %s>' % (self.__class__.__name__, self.domain)

    @classmethod
    def get_or_create(cls, domain, **query_args):
        query_args['domain'] = domain.lower()
        entry = cls.query.filter_by(**query_args).first() or cls(**query_args)
        session.flush()
        return entry

    @classmethod
    def check_match_for_domain(cls, domain, domains=None, **query_args):
        if domains is None:
            domains = cls.domains(**query_args)
        if domains:
            if not domain.startswith('.'):
                domain = '.' + domain
            for list_domain in domains:
                other_domain = list_domain if list_domain.startswith('.') else '.' + list_domain
                if domain.endswith(other_domain):
                    return True
        return False

    @classmethod
    def check_match(cls, url, domains=None, **query_args):
        domain = domain_of_url(url, with_subdomains=True)
        return cls.check_match_for_domain(domain, domains=domains, **query_args)

    @classmethod
    def domains(cls, **query_args):
        return [row.domain for row in cls.query.filter_by(**query_args)]


class DomainGamesList(BlacklistBase, Base):
    __tablename__ = "domain_whitelist" # a misnomer


class DomainTestSiteList(BlacklistBase, Base):
    __tablename__ = "domain_test_sites"


class DomainBlacklist(BlacklistBase, Base):
    __tablename__ = "domain_blacklist"


class UserInitiatedDomain(BlacklistBase, Base):
    __tablename__ = "user_initiated"


class RotatingContentPage(Base): 
    __tablename__ = "rotating_content"
    id = Column(Integer, primary_key=True)
    remote_id = Column(URL, nullable=False)
    remote_id_sha1 = Column(CHAR(40), nullable=False)

    __table_args__ = (UniqueConstraint('remote_id_sha1', name='rotating_content_sha1'),{})

    def __unicode__(self):
        return u'<%s: %s>' % (self.__class__.__name__, self.remote_id)

    @classmethod
    def get_or_create(cls, remote_id):
        remote_id = parse_url(remote_id)
        rm_id_sha1 = sha1(remote_id).hexdigest()
        page = (cls.query.filter_by(remote_id_sha1 = rm_id_sha1).first() or
                cls(remote_id = remote_id, remote_id_sha1 = rm_id_sha1))
        session.flush()
        return page


BLACKLIST_REGEXES = map(re.compile, [
    r'http://www\.last\.fm/ads.php.*',
    r'http://www\.stickam\.com/(joinLive|onlineMembers|preSendFriendRequest|liveStream|largeChatNew|largeChat|joinGroupChat).do.*',
    # Reject ustream unless URL has /recorded (negative look-ahead assertion)
    r'http://www\.ustream\.tv/(?!recorded/\d+)', 
    # must have /b/ or /w/ . 
    r'http://[^.]*\.justin\.tv/(?!.*/[bw]/)',
    # Photos
    r'http://.*photobucket\.com/.*/albums/.*',
    r'http://popminute\.com/photo/.*',
    r'http://www\.filmannex\.com/galleries/.*',
    r'http://www\.moviefone\.com/dvd.*',
    r'http://(www.)?imgur\.com/(r|gallery)/.*',
    # huffpo blog entries have no video
    r'http://www\.huffingtonpost\.com/.*/the-blog',
    # Forum-related stuff
    r'.*showthread\.php.*',
    r'.*/forums?/.*',
    r'https?://(www\.)?forums?\.',
    # Search stuff
    r'.*/\??(register|query|search|find|login|signup)(\.php|\.aspx|\.asp|\.cgi|\.py|=|/|#|\?).*',
    r'http://www\.youtube\.com/watch$',
    # JW Player's ad serving module, used by several different sites
    '.*/modules/video/adtracking/.*',
])


def dont_process_url(url, domain=None, blacklist_domains=None, games_domains=None):
    if not url.startswith('http'):
        return False
    if domain is None:
        domain = domain_of_url(url, with_subdomains=True)
    if DomainBlacklist.check_match_for_domain(domain, domains=blacklist_domains):
        return True
    if DomainGamesList.check_match_for_domain(domain, domains=games_domains):
        return True
    for blacklist_re in BLACKLIST_REGEXES:
        if blacklist_re.match(url):
            return True
    return False
