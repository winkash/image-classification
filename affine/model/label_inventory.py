import time
from datetime import datetime, timedelta
from logging import getLogger
from collections import defaultdict

from affine.model._sqla_imports import *
from affine.model.base import *
from affine.model.web_pages import WebPageInventory
from affine.model.labels import Label
from affine.model.mturk.hits import VideoHit, PageHit
from affine.model.training_data import LabelTrainingPage
from affine.model.web_page_label_results import WebPageLabelResult
from affine import sphinx

logger = getLogger(__name__)
HIT_WEIGHT_CAP = 1.5

__all__ = ['LabelInventory', 'generate_accuracies', 'query_sphinx_forecasting']

class LabelInventory(Base):
    __tablename__ = 'label_inventory'

    id = Column(Integer, primary_key=True)
    created = Column(DateTime, nullable=False, default=datetime.utcnow)
    label_id = Column(Integer, ForeignKey('labels.id'), nullable=False)
    active_pages = Column(Integer, nullable=False)
    opportunity = Column(BigInteger, nullable=False)
    adult_active_pages = Column(Integer, nullable=False)
    filtered_active_pages = Column(Integer, nullable=False)
    filtered_opportunity = Column(BigInteger, nullable=False)
    precision_videocollage = Column(Float)
    precision_screenshot = Column(Float)
    true_positives_videocollage = Column(Integer)
    true_positives_screenshot = Column(Integer)
    hit_count_videocollage = Column(Integer)
    hit_count_screenshot = Column(Integer)
    campaign_count = Column(Integer)
    conflicts_videocollage = Column(Integer)
    conflicts_screenshot = Column(Integer)

    @classmethod
    def get_label_stats(cls, label_id, start_date=datetime(2013,6,1), end_date=None):
        """ Returns a dict with all stats from label-inventory for all timestamps """
        stats = defaultdict(dict)
        if end_date is None:
            end_date = datetime.utcnow()
        all_data = cls.query.filter_by(label_id=label_id).filter(cls.created>=start_date, cls.created<end_date).order_by(cls.created)
        for data in all_data:
            timestamp = datetime.strftime(data.created, '%Y-%m-%d')
            stats[timestamp] = {
                'active_pages' : data.active_pages,
                'opportunity' : data.opportunity,
                'campaign_count' : data.campaign_count,
                'adult_active_pages' : data.adult_active_pages,
                'filtered_active_pages' : data.filtered_active_pages,
                'filtered_opportunity' : data.filtered_opportunity,
                'precision_videocollage' : data.precision_videocollage,
                'precision_screenshot' : data.precision_screenshot,
                'true_positives_videocollage' : data.true_positives_videocollage,
                'true_positives_screenshot' : data.true_positives_screenshot,
                'hit_count_videocollage' : data.hit_count_videocollage,
                'hit_count_screenshot' : data.hit_count_screenshot,
                'conflicts_screenshot' : data.conflicts_screenshot,
                'conflicts_videocollage' : data.conflicts_videocollage,
            }
        return stats

    def __unicode__(self):
        return "<LabelInventory(%s) label_id(%s)>" %(self.id, self.label_id)

def generate_accuracies(label_ids, new_date, old_date):
    """ Returns dict of label_ids along with their accuracies and hits (use only for large number of label_ids!) """
    start_time = time.time()
    screenshot_result = gen_QA_stats(label_ids, new_date, old_date, 'screenshot')
    collage_result = gen_QA_stats(label_ids, new_date, old_date, 'videocollage')
    time_taken = time.time() - start_time
    logger.info('Finished calculating precision and hits in %s seconds' %time_taken)
    return screenshot_result, collage_result


def _gen_mturk_map(label_ids, hit_type, new_date, old_date):
    mturk_map = {}
    mturk_query = session.query(hit_type.label_id, hit_type.page_id, hit_type.result)
    mturk_query = mturk_query.filter(hit_type.timestamp>old_date, hit_type.timestamp<new_date, hit_type.label_id.in_(label_ids))
    mturk_query = mturk_query.filter(hit_type.outstanding==False)
    mturk_query = mturk_query.join(WebPageInventory, WebPageInventory.page_id==hit_type.page_id)
    for label_id, page_id, result in mturk_query:
        if label_id not in mturk_map:
            mturk_map[label_id] = {True:set(), False:set(), None:set()}
        mturk_map[label_id][result].add(page_id)
    return mturk_map


def gen_QA_stats(label_ids, new_date, old_date, result_type):
    if not label_ids:
        return {}
    if result_type == 'videocollage':
        hit_type = VideoHit
    elif result_type == 'screenshot':
        hit_type = PageHit
    else:
        raise ValueError("Unknown result_type")

    mm = _gen_mturk_map(label_ids, hit_type, new_date, old_date)
    results = {}

    # Order label_ids by rank
    labels = Label.query.filter(Label.id.in_(label_ids)).all()
    labels = _sort_labels(labels)
    for l in labels:
        results[l.id] = _get_label_stats_list(l, mm, results, result_type)
    return results


def _get_label_stats_list(label, mturk_map, results, result_type):
    ''' Returns list [prec, tps, hits, conflicts] for label '''
    label_id = label.id
    wplr = WebPageLabelResult
    logger.info("Computing for %s", label_id)
    if label_id not in mturk_map:
        return [0, 0, 0, 0]
    page_ids = mturk_map[
        label_id][True] | mturk_map[label_id][False] | mturk_map[label_id][None]
    training_page_set = set(LabelTrainingPage.get_all_training_page_ids(label_id))
    page_ids = page_ids - training_page_set
    true_page_ids = {p for (p,) in session.query(wplr.page_id).filter(
        wplr.label_id==label_id, wplr.page_id.in_(page_ids))}
    # ignore pages used to train label weights
    true_page_ids = true_page_ids - training_page_set
    tps = len(true_page_ids & mturk_map[label_id][True])
    fps = len(true_page_ids & mturk_map[label_id][False])
    conflicts = len(true_page_ids & mturk_map[label_id][None])
    hits = len((mturk_map[label_id][True] | mturk_map[label_id][False]) - training_page_set)
    prec = round(_calc_prec(tps, tps + fps), 2)

    # Check if the label is not qa-ed and only has child labels
    if _check_qa_label(label, result_type):
            prec = round(_calc_weighted_prec(
                label, results, result_type), 2)
    return [prec, tps, hits, conflicts]


def _sort_labels(labels):
    label_rank = dict((l.id, l.rank) for l in labels)
    sorted_labels = sorted(
        labels,
        key=(lambda label: label_rank[label.id]),
        reverse=True)
    return sorted_labels


def _check_qa_label(l, result_type):
    if l.weighted_labels and l.weighted_keywords == [] and \
            l.weighted_clf_targets == []:
        if (result_type == 'videocollage' and l.qa_enabled is False) or (
                result_type == 'screenshot' and l.page_qa_enabled is False):
            return True
    return False


def _calc_weighted_prec(label, results, result_type):
    """ Calculates weighted-label precision using precisions and filtered-pages
        for child-labels (used for labels which are public but are not qa-enabled)

        Args:
            label: input label
            results: dict of {label_id: [prec, tps, hits, conflicts]}
            result_type: videocollage or screenshot
    """
    total_filt_act_pages = 0.
    weighted_prec = 0.
    child_lids = [i.child_id for i in label.weighted_labels]
    for child_lid in child_lids:
        li = LabelInventory.query.filter_by(label_id=child_lid).order_by(
            LabelInventory.created.desc()).first()
        if not li:
            continue
        total_filt_act_pages += li.filtered_active_pages
        if child_lid in results:
            prec = results[child_lid][0]
        else:
            if result_type == 'videocollage':
                prec = li.precision_videocollage
            elif result_type == 'screenshot':
                prec = li.precision_screenshot
        weighted_prec += li.filtered_active_pages*prec
    weighted_prec = _divide(weighted_prec, total_filt_act_pages)
    return weighted_prec


def _divide(num, den):
    if not den:
        return 0.
    else:
        return float(num)/den

def _calc_prec(trues_right, trues_total):
    if trues_total == 0:
        true_precision = 0.0
    else:
        true_precision = (100. * trues_right) / trues_total
    return true_precision


def query_sphinx_forecasting(label_id):
    adult_id = 994
    pre_roll_id = 9961

    results = [0]*6
    with sphinx.SphinxClient() as client:
        # Pages, Impressions marked for input label and pre-roll label
        query_sphinx = "SELECT COUNT(DISTINCT web_page_id), sum(impressions) FROM \
        forecasting WHERE label_id IN (%(label)s) AND label_id IN (9961) GROUP BY group_by_dummy"
        params = {'label': label_id}
        sphinx_results = client.execute(query_sphinx, params)
        if sphinx_results:
            results[0] = sphinx_results[0][-1]
            results[1] = sphinx_results[0][-2]

        # Pages, Impressions marked for input, adult and pre-roll labels
        query_sphinx = "SELECT COUNT(DISTINCT web_page_id), sum(impressions) FROM \
        forecasting WHERE label_id IN (%(label)s) AND label_id IN (%(adult_label)s) \
        AND label_id IN (9961) GROUP BY group_by_dummy"
        params['adult_label'] = adult_id
        sphinx_results = client.execute(query_sphinx, params)
        if sphinx_results:
            results[2] = sphinx_results[0][-1]
            results[3] = sphinx_results[0][-2]

        # Pages, Impressions for pages
        # AND NOT HAVING labels - adult
        query_sphinx = "SELECT COUNT(DISTINCT web_page_id), sum(impressions) FROM \
        forecasting WHERE label_id IN (%(label)s) AND label_id NOT IN (%(adult_label)s) \
        AND label_id IN (9961)"
        query_sphinx += " GROUP BY group_by_dummy"
        sphinx_results = client.execute(query_sphinx, params)
        if sphinx_results:
            results[4] = sphinx_results[0][-1]
            results[5] = sphinx_results[0][-2]

    return results


def check_inventory_change(label_ids, ratio, day_range):
    """Raise alert if inventory drops or increases over 5x in the past
    day
    """
    from classification_dashboard.utils.label_utils_functions import \
        get_label_inventory_for_date
    today = datetime.utcnow()
    start_day = datetime.utcnow() - timedelta(days=day_range)
    alerts = []
    today_results = get_label_inventory_for_date(label_ids, today)
    start_results = get_label_inventory_for_date(label_ids, start_day)
    for l_id in label_ids:
        today_inv = today_results[l_id].pages
        start_inv = start_results[l_id].pages
        if today_inv is not None and start_inv is not None:
            rate = abs(today_inv - start_inv)*100 / start_inv
            if today_inv > ratio * start_inv:
                alerts.append(
                    'Label %s has an increase of inventory by %s%% over '
                    'last %s days.' % (l_id, rate,  day_range))
            elif today_inv < start_inv / ratio:
                alerts.append(
                    'Label %s has a decrease of inventory by %s%% over '
                    'last %s days.' % (l_id, rate, day_range))

    return alerts
