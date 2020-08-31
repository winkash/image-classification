from Pycluster import Tree, Node, treecluster
from itertools import combinations
import hashlib
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import numpy as np
import cv
import cv2
from affine.detection.model.util import PickleMixin
from logging import getLogger

__all__ = ['PyLinkage', 'AbstractClustering', 'Grouping', 'Cluster', 'chisqr',
           'intersection', ]

logger = getLogger("affine.detection.clustering")


class PyLinkage(object):

    @staticmethod
    def convert_tree_to_mat(tree):
        """ Converts a tree to a linkage matrix
            Args:
                tree : Tree data sctructure as given by
                Pycluster.treeclassifier
            Returns:
                linkage : a linkage matrix as given by the
                scipy.cluster.hierarchy.linkage method
        """
        mat = []
        sizes = {}
        index = len(tree) + 1
        for i, node in enumerate(tree):
            size = 0
            if node.left < 0:
                left = abs(node.left) + len(tree)
                size += sizes[left]
            else:
                left = node.left
                size += 1
            if node.right < 0:
                right = abs(node.right) + len(tree)
                size += sizes[right]
            else:
                right = node.right
                size += 1
            sizes[index] = size
            index += 1
            left, right = min(left, right), max(left, right)
            mat.append([left, right, node.distance, size])

        return np.asarray(mat)

    @staticmethod
    def convert_condensed_to_lower(dist_mat):
        """ Converts a scipy condensed matrix into a lower triangular list
        of lists
            Args:
                dist_mat : condensed distance matrix
                (o/p of scipy.spatial.distance.pdist)
            Returns :
                List of lists where each list corresponds to
                the row of the lower traingular distance matrix
                as required by the Pycluster module
        """
        assert len(dist_mat) > 1, \
            "Must have atleast more than 1 element in given dist_mat"
        n = int(np.ceil(np.sqrt(2 * len(dist_mat))))
        idx = 0
        mat = [[] for i in range(n)]
        for col in range(0, n):
            for row in range(0, n):
                if row > col:
                    mat[row].append(dist_mat[idx])
                    idx += 1
        return mat

    @classmethod
    def compute_linkage(cls, conden_dist_mat):
        lower_dist_mat = cls.convert_condensed_to_lower(conden_dist_mat)
        tree = treecluster(distancematrix=lower_dist_mat, method='s')
        return cls.convert_tree_to_mat(tree)


class AbstractClustering(PickleMixin):

    def __init__(self, distance, method, min_occurence=3):
        self.distance = distance
        self.method = method
        self.min_occurence = min_occurence
        self.distance_matrix = []
        self.all_clusters = {}
        self.clusterings = []

    def _cleanse_dist_mat(self):
        for idx, val in enumerate(self.conden_dist_mat):
            if val < 0:
                self.conden_dist_mat[idx] = 0

    def compute_distance(self, data):
        """ Computes the condensed distance matrix
            Args:
                data: n x m numpy array where n is the number of features
                and m is the number of dimensions
            Raises:
                ValueError if distance is not supported
        """
        logger.info('Computing Distances')
        self.num_points = len(data)
        # TODO: Maybe make this into a single mapping function and remove elif
        # ladder
        if self.distance == 'mahalanobis':
            data = np.exp(-1 * data / data.std())
            self.conden_dist_mat = pdist(data, self.distance)
        elif self.distance in ['euclidean', 'cosine']:
            self.conden_dist_mat = pdist(data, self.distance)
        elif self.distance == 'chisqr':
            self.conden_dist_mat = pdist(data, chisqr)
        elif self.distance == 'intersection':
            self.conden_dist_mat = pdist(data, intersection)
        else:
            raise ValueError("distance type not supported")
        self._cleanse_dist_mat()

    def compute_linkage(self):
        logger.info('Computing Linkage in scipy')
        dist_mat = self.conden_dist_mat
        if self.method == 'ward' and len(dist_mat.shape) == 1:
            dist_mat = squareform(dist_mat)
        return linkage(dist_mat, method=self.method)

    @staticmethod
    def cluster_indices(clusters):
        """ Gets indices of non singleton clusters
            Args:
                clusters: array of indices that indicate cluster labels
                for all points
            Returns:
                indices: dictionary of labels, where the values are the
                position of the points
        """
        indices = {}
        data = np.array(range(len(clusters)))
        counts = Counter(clusters)
        for c, v in counts.items():
            if v > 1 and v < len(clusters):
                indices[c] = data[clusters == c]
        return indices

    def hierarchical_clustering(self):
        """ Hierarchical clustering. Computes statistics about the clusters,
        such as minimum variance, silhouettem and ocurrence
        """
        eps = 0.01
        logger.info("computing hierarchical clustering with distance " +
                    "%s and method %s" % (self.distance, self.method))
        lnk = self.compute_linkage()
        thresholds = self.choose_thresholds(lnk)

        # TODO: Try and make this parallel
        for th in thresholds:
            clusters = fcluster(lnk, th - eps, 'distance')
            indices = self.cluster_indices(clusters)
            sliceclustering = hashlib.sha1(clusters).hexdigest()
            newclustering = Grouping(sliceclustering, clusters, th)
            self.clusterings.append(newclustering)
            logger.debug('found clusters for threshold %f' % th)
            for idx in indices.values():
                clu = hashlib.sha1(idx).hexdigest()
                if clu not in self.all_clusters:
                    newcluster = Cluster(clu, idx)
                    newcluster.add_clusterings(sliceclustering)
                    self.all_clusters[clu] = newcluster
                else:
                    self.all_clusters[clu].count += 1
                    self.all_clusters[clu].add_clusterings(sliceclustering)


class Grouping(object):

    def __init__(self, name, labels, threshold):
        self.labels = labels
        self.name = name
        self.threshold = threshold


class Cluster(object):

    def __init__(self, name, indices):
        self.indices = indices
        self.count = 1
        self.size = len(indices)
        self.name = name
        self.clustering = []
        self.minvar = -1
        self.avg_dist = None

    def add_clusterings(self, clustering):
        """ adds to the list of groupings where this cluster was generated
            Args:
                clustering: a list of strings, where the string is the name
                of the clustering
        """
        self.clustering.append(clustering)

    def get_stats(self):
        """ Returns a string with the most important stats of a cluster
            Returns:
                stats: a string of statistics about the cluster
        """
        stats = \
            'cluster: %s\ncount = %d, size = %d, minvar = %f, avg_dist = %s\n'\
            % (self.name, self.count, self.size, self.minvar, self.avg_dist)
        return stats

    @classmethod
    def are_clusters_similar(cls, c1, c2, proportion=0.8):
        """ Compares 2 clusters and returns name of smaller cluster
            if it is similar to the bigger one. 2 clusters are similar if
                a. the samller cluster is a subset of the bigger cluster
                b. the 2 clusters are similar in size and their intersection
                   is more than proportion times len of smaller cluster
            Args:
                c1, c2 : instances of the Cluster class
                proportion : value representing how much common
                             should be present between the two clusters
            Returns:
                The name of the smaller cluster if the two are similar else
                None
        """
        if len(c1.indices) > len(c2.indices):
            c1_idx = set(c1.indices.tolist())
            c2_idx = set(c2.indices.tolist())
            smaller = c2.name
        else:
            c1_idx = set(c2.indices.tolist())
            c2_idx = set(c1.indices.tolist())
            smaller = c1.name

        if (len(c1_idx & c2_idx) > proportion * len(c2_idx) and
                len(c2_idx) >= 0.8 * len(c1_idx)) or\
                len(c1_idx & c2_idx) == len(c2_idx):
            return smaller

    @staticmethod
    def get_dist(dist, i, j):
        """ Gets the distance for two points (i and j) from the condensed
        matrix such that
            for a square matrix (n x n) it would correspond to ith row and
            jth col
            Args:
                dist : condensed version of the distance matrix
                i : row value
                j : col value
        """
        n = int(np.ceil(np.sqrt(2 * len(dist))))
        if i == j:
            return 0
        elif i > j:
            return dist[n * j - j * (j + 1) / 2 + i - 1 - j]
        else:
            return dist[n * i - i * (i + 1) / 2 + j - 1 - i]

    @classmethod
    def get_all_distances(cls, indices, dist_mat):
        """ Gets list of distances between combiantions of given points
            Args:
                indices : list of indeices (position of points)
                dist_mat : distance/metric matrix (condensed version)
                n : total number of points
        """
        distances = []
        for i, j in combinations(indices, 2):
            distances.append(cls.get_dist(dist_mat, i, j))
        return distances

    @classmethod
    def get_cluster_average(cls, indices, dist_mat):
        """ Computes the average of distances for given set of indices
            Args:
                dist_mat: distance/metric matrix (condensed version)
                indices: list of indices (position of points in feature array)
                n : total number of points
            Returns:
                var: average of distance for the points
        """
        distances = cls.get_all_distances(indices, dist_mat)
        return np.mean(distances)

    @classmethod
    def get_cluster_variance(cls, indices, dist_mat):
        """ Computes the variance of distances for given set of indices
            Args:
                dist_mat: distance/metric matrix (condensed version)
                indices: list of indices (position of points in feature array)
                n : total number of points
            Returns:
                var: variance of distance for the points
        """
        distances = cls.get_all_distances(indices, dist_mat)
        return np.var(distances)


def chisqr(p1, p2):
    return _cv_dist(p1, p2, 'chisqr')


def intersection(p1, p2):
    return _cv_dist(p1, p2, 'intersection')


def _cv_dist(p1, p2, distance):
    distance_values = {'chisqr': cv.CV_COMP_CHISQR,
                       'intersection': cv.CV_COMP_INTERSECT}
    assert distance in distance_values, "distance should be amongst %s got %s" \
        % (','.join(distance_values.keys()), distance)
    p1 = p1.astype('float32')
    p2 = p2.astype('float32')
    return cv2.compareHist(p1, p2, distance_values[distance])
