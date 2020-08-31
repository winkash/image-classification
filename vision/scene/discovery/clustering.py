import numpy as np
from logging import getLogger
from affine.detection.model.clustering import *

__all__ = ['SceneClustering']

logger = getLogger(__name__)

class SceneClustering(AbstractClustering):
    def choose_thresholds(self, linkage):
        """ Chooses thresholds where two or more clusters where merged
            Args:
                lnk: numpy matrix (n x 4) that is the result from linkage. N is the number of clusterings
            Returns:
                thresholds: a list of thresholds (float)
            Assertions:
                AssertError when lnk has the incorrect shape

        """
        assert linkage.shape[1] == 4, 'a linkage matrix is needed to choose thresholds'
        sz = np.mean(linkage[:, 3])
        thresholds = linkage[linkage[:, 3] > sz, 2]
        thresholds = np.unique(thresholds)
        thresholds = np.sort(thresholds)[::-1]
        logger.info('Obtained %d thresholds', len(thresholds))
        return thresholds

    def select_clusters(self):
        """ Selects a set of clusters that meet a certain criteria, such as
            occurrence of clusters > 2, size of cluster < half of the images and > average size
        """
        logger.info("selecting clusters")
        assert len(self.all_clusters), 'clustering results not found'
        self.chosen = []
        minvar = []
        med = self.num_points / 2.0
        medsize = np.mean([clu.size for clu in self.all_clusters.values()])
        for k, cluster in self.all_clusters.items():
            if cluster.count < self.min_occurence or cluster.size > med or cluster.size < medsize:
                self.all_clusters.pop(k, None)
            else:
                cluster.minvar = Cluster.get_cluster_variance(cluster.indices, self.conden_dist_mat)
                minvar.append(cluster.minvar)
        if minvar:
            meanvar = np.mean(minvar)
            chosen = [clu for clu in self.all_clusters.values() if clu.minvar <= meanvar]
            self.chosen.sort(key=lambda k:(-k.size, -k.count))



