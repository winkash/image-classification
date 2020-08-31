import itertools
import pickle
from logging import getLogger

from scipy.stats import chisquare
from affine.detection.model.classifiers import *
from affine.detection.vision.facerec import *
from affine.parallel import pool_map

__all__ = ['FaceIndexAnalyzer']

logger = getLogger('affine.detection.vision.facerec.face_index_analyzer')

MAX_P_VALUE = 0.1

class FaceIndexAnalyzer(object):
    def __init__(self, face_index, train_boxes, train_labs, test_boxes, test_labs=None, negative_boxes=None, is_built=False):
        self.train_boxes = train_boxes
        self.train_labs = train_labs
        self.test_boxes = test_boxes
        self.test_labs = test_labs
        self.face_index = face_index
        self.negative_boxes = negative_boxes
        self.nearest_neighbor = {}
        self._index_built = is_built
        self.neighbors = []
        self.neg_neighbors = []

    def radius_range(self):
         return [i for i in range(16, 22)]

    def prob_delta_range(self):
        return [i/10. for i in range(1, 6)]

    def k_range(self):
        return [11, 15, 21, 27]

    def build_index(self):
        """Checks if the index is built and if not builds the index
        """
        if self._index_built == False:
            logger.info('Starting to build index')
            self.face_index.build_index(self.train_boxes, self.train_labs)
            self._index_built = True

    def get_actual_neighbors(self):
        """Gets closest neighbor using brute force NN
        Used to calculate selectivity and recallfor indexing
        """
        knn = KNNScikit(neighbors=1)
        train_feats = self.face_index.get_signatures(self.train_boxes)
        test_feats = self.face_index.get_signatures(self.test_boxes)

        knn.train(train_feats, self.train_labs)
        res = knn.knc.kneighbors(test_feats, return_distance=False)
        for box_id, r in zip(self.test_boxes, res):
            self.nearest_neighbor[box_id] = self.train_boxes[r[0]]

    def get_neighbors(self):
        """Get neighbors for all test boxes
        """
        if self.neighbors == []:
            for box_id in self.test_boxes:
                self.neighbors.append(self.face_index.get_neighbors(box_id))

    def get_negative_neighbors(self):
        """Get neighbors for all negative boxes
        """
        if self.neg_neighbors == []:
            for box_id in self.negative_boxes:
                self.neg_neighbors.append(self.face_index.get_neighbors(box_id))

    def analyze(self):
        """Get Recall and Selectivity values for an index,
        Recall : if the potential neighbors have the nearest neighbor
        Sel : Avg no of potential neighbors that we need to perform Brute KNN
        """
        logger.info('Calculating Recall and Selectivity')
        self.build_index()
        self.get_actual_neighbors()
        sel = 0.0
        recall = 0

        self.get_neighbors()

        for nn, box_id in zip(self.neighbors, self.test_boxes):
            near_boxes = [r[0] for r in nn]
            if self.nearest_neighbor[box_id] in near_boxes:
                recall += 1
            sel += len(nn)
        recall = 100.*recall/len(self.test_boxes)
        sel = float(sel)/len(self.test_boxes)
        return sel, recall

    def analyze_buckets(self):
        """ Analyze KMeansHashing bucket distribution """
        logger.info('Analyzing bucket distribution')
        lsh = self.face_index.lsh
        assert lsh.hash_type == 'kmeans', "Analysis valid only for kmeans hashing"
        tables = lsh.hash_tables
        for ht in tables: #each hashtable is type HashTableBase KMeansHashing
            distrib = ht.distrib #ndarray
            chisq, p = chisquare(distrib)
            if p < MAX_P_VALUE:
                return False
        return True

    def get_tp_rate(self, r, p, k):
        """Get the true positive rate for given (radius, prob_delta, K) combination
        Also returns the true negatives for further analysis
        """
        tp = 0
        tns = []
        self.get_neighbors()
        for box_id, nn, act_lab in zip(self.test_boxes, self.neighbors, self.test_labs):
            res, ll = self.face_index.get_verdict(nn, radius=r, prob_delta=p, k=k)
            if res == act_lab:
                tp += 1
            else:
                tns.append(BadBox(box_id, nn[:k], res, act_lab, ll))
        return 100.*tp/len(self.test_boxes), tns

    def get_fp_rate(self, r, p, k):
        """Get False Positive Rate for given (radius, prob_delta, K) combination
        Also returns false positives
        """
        fp = 0
        fps = []
        self.get_negative_neighbors()
        for box_id, nn in zip(self.negative_boxes, self.neg_neighbors):
            res, ll = self.face_index.get_verdict(nn, radius=r, prob_delta=p, k=k)
            if res != None:
                fp += 1
                fps.append(BadBox(box_id, nn[:k], res, None, ll))
        return 100.*fp/len(self.negative_boxes), fps

    def get_tpfp_rates(self, **kwargs):
        """Get True Positive Rates for all combinations of (radius, prob_delta, k)
        """
        logger.info('Evaluating for TP/FP Rates...')
        assert self.test_labs
        assert self.negative_boxes

        self.get_neighbors()
        self.get_negative_neighbors()
        results = []
        for r, p, k in itertools.product(self.radius_range(), self.prob_delta_range(), self.k_range()):
            logger.info('Evaluating for (r, p, k): %f, %f, %f' %(r, p, k))
            tp, _ = self.get_tp_rate(r, p, k)
            fp, _ = self.get_fp_rate(r, p, k)
            logger.info("Results tp / fp : %f / %f" %(tp, fp))
            results.append([r, p, k, tp, fp])

        return results

    def save_to_file(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        logger.info('Successfully saved the analyzer to %s' %file_path)

    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, 'rb') as f:
            fia = pickle.load(f)
        return fia


class BadBox(object):
    def __init__(self, box_id, neighbors, res, lab, likelihood=None):
        self.box_id = box_id
        self.nn = neighbors
        self.res = res
        self.actual_lab = lab
        self.ll = likelihood


"""
Utils that are used to provide parallel processing for face index analysis
"""
def get_neighbor_single_box(args):
    face_index, box_id = args
    return face_index.get_neighbors(box_id)
