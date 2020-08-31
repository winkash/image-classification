from itertools import cycle, islice

from affine.model._sqla_imports import *
from affine.model.base import session
from affine.model.mturk.hits import GoldenHit, GoldenHitCandidate, get_hit_from_hit_id
from affine.model.mturk.evaluators import MechanicalTurkEvaluator

__all__ = ['GoldenHitManager']

class GoldenHitManager(object):
    """Manager in charge of golden hits."""

    @staticmethod
    def submit_golden_hits(n_hits, n_lookback):
        """Submit golden hits.

        Fetches the N_LOOKBACK hits most recently selected for golden submission
        and submits N_HITS of them, cycling through them as necessary, and
        prioritizing those that have been submitted as golden the least number of
        times.

        Args:
            n_hits: Number of golden hits submissions.
            n_lookback: Number of distinct hits used for submission.

        Raises:
            AssertionError: No candidate golden hits
        """
        query = session.query(GoldenHitCandidate.hit_id)
        assert query.count() > 0, "No candidate golden hits"
        query = query.order_by(GoldenHitCandidate.created_at.desc()).limit(n_lookback)
        query = query.from_self()
        query = query.outerjoin(GoldenHit, GoldenHitCandidate.hit_id == GoldenHit.hit_id)
        query = query.group_by(GoldenHitCandidate.hit_id)
        query = query.order_by(func.count(GoldenHit.hit_id).asc()).limit(n_hits)
        hit_ids = islice(cycle([hit_id for (hit_id,) in query]), n_hits)
        for hit in map(get_hit_from_hit_id, hit_ids):
            ghid = MechanicalTurkEvaluator.create_duplicate_hit(hit)
            GoldenHit(golden_hit_id=ghid, hit_id=hit.hit_id)
        session.flush()

    @staticmethod
    def get_potential_candidates(counts):
        """Get completed hits that could be good golden hits.

        Args:
            counts: List of tuples (hit class, number of hits for the class)

        Returns:
            List of hits.
        """
        hits = []
        for cls, count in counts:
            query = cls.get_potential_golden_hit_candidates()
            query = query.outerjoin(GoldenHitCandidate,
                                    cls.hit_id == GoldenHitCandidate.hit_id)
            query = query.filter(GoldenHitCandidate.hit_id == None).limit(count)
            hits.extend(query.all())
        return hits
