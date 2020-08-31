from logging import getLogger
from itertools import combinations

from affine.detection.model.clustering import *
from affine.detection.model.util import PickleMixin

__all__ = ['FaceCluster', 'FaceClustering']

logger = getLogger(__name__)

class FaceCluster(PickleMixin):
    def __init__(self, clusters, data_bag, store_feats=False):
        super(FaceCluster, self).__init__()
        self.clusters = clusters
        self.data_bag = data_bag
        if store_feats == False:
            self.data_bag.feats = None


class FaceClustering(AbstractClustering):

    def __init__(self, distance, method, min_size, max_size, min_occurence_count, min_avg_dist):
        assert method == 'single', "Only mclustering method 'single' is allowed for FaceClustering"
        super(FaceClustering, self).__init__(distance, method, min_occurence_count)
        self.min_size = min_size
        self.max_size = max_size
        self.min_occurence_count = min_occurence_count
        self.min_avg_dist = min_avg_dist

    def compute_linkage(self):
        logger.info('Computing Linkage using PyCluster')
        return PyLinkage.compute_linkage(self.conden_dist_mat)

    def select_clusters(self):
        logger.info('Performing cluster selection')
        for k, cluster in self.all_clusters.items():
            if cluster.count < self.min_occurence_count or cluster.size < self.min_size or cluster.size >= self.max_size:
                self.all_clusters.pop(k, None)

        logger.info('Calculating Cluster metrics')
        for name, cluster in self.all_clusters.items():
            cluster.avg_dist = Cluster.get_cluster_average(cluster.indices, self.conden_dist_mat)

        self.chosen_ones = {}
        for name, cluster in self.all_clusters.items():
            if cluster.avg_dist <= self.min_avg_dist:
                self.chosen_ones[name] = cluster

        self.remove_similar_clusters()

    def choose_thresholds(self, linkage):
        # Choose all thresholds where the new cluster obtained
        # is greater than min allowed cluster sizes
        thresholds = {row[2] for row in linkage if row[3] >= self.min_size and row[3] < self.max_size}
        logger.info('Obtained %d thresholds', len(thresholds))
        return thresholds

    def remove_similar_clusters(self):
        for c1, c2 in combinations(self.chosen_ones.keys(), 2):
            if c1 in self.chosen_ones and c2 in self.chosen_ones:
                cluster_to_remove = Cluster.are_clusters_similar(self.chosen_ones[c1], self.chosen_ones[c2])
                if cluster_to_remove:
                    self.chosen_ones.pop(cluster_to_remove, None)
