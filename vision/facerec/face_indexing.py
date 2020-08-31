import math
import os
import pickle
import cv
import numpy as np
import os.path as path

from threading import Lock
from collections import defaultdict
from datetime import datetime
from logging import getLogger
from scipy.spatial.distance import cdist
from scipy.stats import histogram
from scipy.cluster.vq import kmeans2, vq

from affine import config
from affine.aws import s3client
from affine.model import *
from affine.detection.model.projections import PCA, LDA, DLA
from affine.detection.vision.facerec import dynamodb
from affine.parallel import pool_map

logger = getLogger("affine.face_indexing")

FEATURE_LENGTH = 1937

__all__ = ['HashTableBase', 'HammingHashTable', 'EuclideanHashTable',
           'KMeansHashing', 'BoxInfo', 'LSH', 'AbstractSigExtractor',
           'PCAExtractor', 'SubspaceExtractor', 'LDAExtractor',
           'DLAExtractor', 'FaceIndex']

class BoxInfo(object):
    def __init__(self, box_id, feature, label_id):
        self.box_id = box_id
        self.feature = feature
        self.label_id = label_id


class HashTableBase(object):
    def __init__(self, hash_size, feature_size, **kw_args):
        self.hash_size = hash_size
        self.feature_size = feature_size
        self.lookup = defaultdict(list)

    def insert(self, box_info):
        '''generates the hash for given box_info and updates the lookup'''
        feature_hash = self.get_hash(box_info.feature)
        self.lookup[feature_hash].append((box_info.box_id, box_info.label_id))

    def query(self, feature):
        '''query lookup to get near boxes for given feature'''
        key = self.get_hash(feature)
        return self.lookup[key]

    def encode(self):
        return self.__dict__

    @classmethod
    def decode(cls, hash_dict):
        hash_table = cls.__new__(cls)
        hash_table.__dict__ = hash_dict
        return hash_table


class HammingHashTable(HashTableBase):
    def __init__(self, hash_size, feature_size, planes=None, **kw_args):
        super(HammingHashTable, self).__init__(hash_size, feature_size, **kw_args)
        if planes is not None:
            if type(planes) != np.ndarray:
                raise TypeError("planes should be of type 'ndarray'")
            else:
                assert planes.shape == (self.hash_size, self.feature_size), "Given planes do not match size requirements. Expected : (%s, %s)" %(self.hash_size, self.feature_size)
                self.planes = planes
        else:
            self.planes = self._generate_planes()

    def _generate_planes(self):
        return np.random.randn(self.hash_size, self.feature_size)

    def get_hash(self, feature):
        '''returns hash for given feature'''
        if len(feature.shape) == 2:
            if feature.shape[0] > 1:
                raise ValueError('expected single row of feature, got more than one. feature.shape : %s' %str(feature.shape))
            else:
                feature = feature[0]

        feature_hash = ''
        assert self.planes.shape[1] == feature.shape[0], "Planes and features differ in size, Planes : %s, feature %s" %(self.planes.shape, feature.shape)
        for i in np.dot(self.planes, feature):
            feature_hash += str(int(i>=0))
        return feature_hash


class EuclideanHashTable(HashTableBase):
    def __init__(self, hash_size, feature_size, b=0.0, w=4.0, planes=None, **kw_args):
        super(EuclideanHashTable, self).__init__(hash_size, feature_size, **kw_args)
        assert 0 <= b < w, "b must satisfy [0, w), got %f" %b
        self.b = b
        self.w = w
        if planes is not None:
            if type(planes) != np.ndarray:
                raise TypeError("planes should be of type 'ndarray'")
            else:
                assert planes.shape == (self.hash_size, self.feature_size), "Given planes do not match size requirements. Expected : (%s, %s)" %(self.hash_size, self.feature_size)
                self.planes = planes
        else:
            self.planes = self.generate_normal_planes()

    def generate_normal_planes(self):
        return np.random.standard_normal((self.hash_size, self.feature_size))

    def get_hash(self, feature):
        '''returns hash for given feature'''
        if len(feature.shape) == 2:
            if feature.shape[0] > 1:
                raise ValueError('expected single row of feature, got more than one. feature.shape : %s' %str(feature.shape))
            else:
                feature = feature[0]

        feature_hash = ''
        assert self.planes.shape[1] == feature.shape[0], "Planes and features differ in size, Planes : %s, feature %s" %(self.planes.shape, feature.shape)
        for i, val in enumerate(np.dot(self.planes, feature)):
            if i >0:
                feature_hash += ':'
            k = int(math.floor((val+self.b)/self.w))
            feature_hash += str(k)
        return feature_hash


class KMeansHashing(HashTableBase):
    def __init__(self, codebook, distrib):
        hash_size = 1
        feature_size = codebook.shape[1]
        super(KMeansHashing, self).__init__(hash_size, feature_size)
        self.codebook = codebook
        self.distrib = distrib

    def get_hash(self, feature):
        feature = np.reshape(feature, (1, self.codebook.shape[1]))
        code, dist = vq(feature, self.codebook)
        feature_hash = str(code[0])
        return feature_hash

    @classmethod
    def create_hash_tables(cls, num_tables, features, num_clusters,
                           iterations=5):
        hash_tables = []
        codebooks = get_n_codebooks(features, num_tables, num_clusters,
                                    iterations)
        for cb, db in codebooks:
            hash_tables.append(KMeansHashing(cb, db))
        return hash_tables


def _quantize(features, num_clusters, iterations):
    logger.info('Obtaining codebook')
    cb, neigh = kmeans2(features, num_clusters, iter=iterations, minit='points')
    logger.info('Finished quantizing')
    dist, _ , _ , _ = histogram(neigh, numbins=num_clusters)
    return cb, dist

def quantize(args):
    return _quantize(*args)

def get_n_codebooks(features, num_codebooks, num_clusters, iterations):
    codebooks = []
    args = (features, num_clusters, iterations)
    codebooks = pool_map(quantize, args=[args]*num_codebooks)
    return codebooks


class LSH(object):
    def __init__(self, num_hashtables, hash_size, hash_type='hamming', hash_tables=None,  **kw_args):
        self.num_hashtables = num_hashtables
        self.hash_size = hash_size
        self.hash_type = hash_type
        self.hash_tables = hash_tables or []
        if self.hash_type == 'kmeans':
            assert self.hash_tables != [], "For Kmeans hashing, hash tables are required"
        self.kw_args = kw_args

    def _create_hashtables(self, feature_size):
        for i in range(self.num_hashtables):
            if self.hash_type == 'hamming':
                ht = HammingHashTable(self.hash_size, feature_size, **self.kw_args)
            elif self.hash_type == 'euclidean':
                ht = EuclideanHashTable(self.hash_size, feature_size, **self.kw_args)
            else:
                raise ValueError('Invalid hashing type, %s' %self.hash_type)
            self.hash_tables.append(ht)

    def insert(self, box_info):
        '''insert a box into all the hash tables'''
        for ht in self.hash_tables:
            ht.insert(box_info)

    def index(self, boxes_info):
        """ boxes_info : Can be an instance or a list of BoxInfo,
            This method is used to either create or update an existing index
        """
        if not isinstance(boxes_info, list):
            if not isinstance(boxes_info, BoxInfo):
                raise TypeError("Can only index type BoxInfo")
            else:
                boxes_info = [boxes_info]
        else:
            if not isinstance(boxes_info[0], BoxInfo):
                raise TypeError("Can only index type BoxInfo")

        t1 = datetime.utcnow()
        if self.hash_tables == []:
            feature_size = boxes_info[0].feature.shape[0]
            self._create_hashtables(feature_size)

        for i, b in enumerate(boxes_info):
            self.insert(b)
        logger.debug('Finished indexing for %d faces in %s ' %(len(boxes_info), str(datetime.utcnow()-t1)))

    def query(self, feature):
        '''query for a test feature to get near boxes'''
        near_boxes = set()
        for ht in self.hash_tables:
            near_boxes.update(ht.query(feature))
        return list(near_boxes)

    def encode(self):
        lsh_dict = self.__dict__.copy()
        lsh_dict['hash_tables'] = []
        for hash_table in self.hash_tables:
            lsh_dict['hash_tables'].append(hash_table.encode())
        return lsh_dict

    @classmethod
    def decode(cls, lsh_dict):
        hash_tables = []
        for hash_table in lsh_dict['hash_tables']:
            if lsh_dict['hash_type'] == 'kmeans':
                hash_tables.append(KMeansHashing.decode(hash_table))
            elif lsh_dict['hash_type'] == 'hamming':
                hash_tables.append(HammingHashTable.decode(hash_table))
            elif lsh_dict['hash_type'] == 'euclidean':
                hash_tables.append(EuclideanHashTable.decode(hash_table))
            else:
                raise ValueError('Invalid hashing type, %s' % hash_type)
        lsh_dict['hash_tables'] = hash_tables
        lsh = cls.__new__(cls)
        lsh.__dict__ = lsh_dict
        return lsh


class AbstractSigExtractor(object):

    def __init__(self):
        self.face_signature_client = dynamodb.DynamoFaceSignatureClient()

    def extract(self, box_ids):
        ''' this is the wrapper to get the final signature used for hashing,
        it expectes signature to be present in the DB for the given box_id
        and return the projected version based on the projection_type
        '''
        feats = self.face_signature_client.get_signatures(box_ids, raise_exp=True)
        return self.project_signature(np.asarray(feats))

    @classmethod
    def decode(cls, sig_dict):
        extractor_type = sig_dict['extractor_type']
        sig_dict['face_signature_client'] = dynamodb.DynamoFaceSignatureClient()
        if extractor_type == 'dla_extractor':
            return DLAExtractor.decode(sig_dict)
        elif extractor_type == 'pca_extractor':
            return PCAExtractor.decode(sig_dict)
        elif extractor_type == 'lda_extractor':
            return LDAExtractor.decode(sig_dict)
        else:
            raise ValueError('Invalid extractor type, %s' % extractor_type)


class PCAExtractor(AbstractSigExtractor):
    PCA_TAR_BALL_NAME = 'pca'
    def __init__(self):
        super(PCAExtractor, self).__init__()
        pca_file = self.grab_projection_files()
        self.pca = PCA()
        self.pca.load_from_xml_file(pca_file)

    def grab_projection_files(self):
        download_dir = os.path.join(config.scratch_detector_path(), self.PCA_TAR_BALL_NAME)
        bucket = config.s3_detector_bucket()
        logger.info('Downloading files from s3')
        s3client.download_tarball(bucket, self.PCA_TAR_BALL_NAME, download_dir)
        pca_file =  os.path.join(download_dir, 'pca.xml')
        assert os.path.exists(pca_file), pca_file
        return pca_file

    def project_signature(self, feats):
        return self.pca.project(feats)

    def encode(self):
        pca_dict = {}
        pca_dict['extractor_type'] = 'pca_extractor'
        pca_dict['pca'] = self.pca.encode()
        return pca_dict

    @classmethod
    def decode(cls, pca_dict):
        pca_dict['pca'] = PCA.decode(pca_dict['pca'])
        pca_extractor = cls.__new__(cls)
        pca_extractor.__dict__  = pca_dict
        return pca_extractor


class SubspaceExtractor(AbstractSigExtractor):
    def __init__(self, model_dir):
        super(SubspaceExtractor, self).__init__()
        self.model_dir = model_dir

        self.pca_file = os.path.join(self.model_dir, 'learned_pca.xml')
        assert os.path.exists(self.pca_file), self.pca_file
        self.pca = PCA()
        self.pca.load_from_xml_file(self.pca_file)

        self.subspace_file = os.path.join(self.model_dir, 'learned_subspace.xml')
        assert os.path.exists(self.subspace_file), self.subspace_file
        self.get_projection_matrix()

    def get_projection_matrix(self):
        WI = np.asarray(cv.Load(self.subspace_file, name='WI'))
        WI_wh = np.asarray(cv.Load(self.subspace_file, name='WI_wh'))
        WL = np.asarray(cv.Load(self.subspace_file, name='WL'))
        self.projection_mat = np.dot(WL, np.dot(WI_wh, WI)).transpose()

    def project(self, feats):
        return np.dot(feats, self.projection_mat)

    def project_signature(self, feats):
        red_feats = self.pca.project(feats)
        return self.project(red_feats)


class LDAExtractor(AbstractSigExtractor):
    def __init__(self, pca_file, lda_file):
        super(LDAExtractor, self).__init__()
        self.lda = LDA.load_from_file(lda_file)
        self.pca = PCA()
        self.pca.load_from_xml_file(pca_file)

    def project_signature(self, feats):
        pca_feats = self.pca.project(feats)
        return self.lda.project(pca_feats, whiten=True)

    def encode(self):
        dla_dict = {}
        dla_dict['extractor_type'] = 'lda_extractor'
        dla_dict['lda'] = self.lda.encode()
        dla_dict['pca'] = self.pca.encode()
        return dla_dict

    @classmethod
    def decode(cls, lda_dict):
        lda_dict['pca'] = PCA.decode(lda_dict['pca'])
        lda_dict['lda'] = LDA.decode(lda_dict['lda'])
        lda_extractor = cls.__new__(LDAExtractor)
        lda_extractor.__dict__  = lda_dict
        return lda_extractor


class DLAExtractor(AbstractSigExtractor):
    def __init__(self, pca_file, dla_file):
        """
        Args:
            pca_file : the xml file that is saved as part of PCA learning
            dla_file : pickle file storing the dla
        """
        super(DLAExtractor, self).__init__()
        self.dla = DLA.load_from_file(dla_file)
        self.pca = PCA()
        self.pca.load_from_xml_file(pca_file)

    def project_signature(self, feats):
        pca_feats = self.pca.project(feats)
        return self.dla.project(pca_feats)

    def encode(self):
        dla_dict = {}
        dla_dict['extractor_type'] = 'dla_extractor'
        dla_dict['dla'] = self.dla.encode()
        dla_dict['pca'] = self.pca.encode()
        return dla_dict

    @classmethod
    def decode(cls, dla_dict):
        dla_dict['pca'] = PCA.decode(dla_dict['pca'])
        dla_dict['dla'] = DLA.decode(dla_dict['dla'])
        dla_extractor = cls.__new__(DLAExtractor)
        dla_extractor.__dict__  = dla_dict
        return dla_extractor


class FaceIndex(object):
    def __init__(self, hash_size, num_hashtables, sig_extractor, **kw_args):
        self.hash_size = hash_size
        self.num_hashtables = num_hashtables
        self.sig_extractor = sig_extractor
        self.lsh = LSH(self.num_hashtables, self.hash_size, **kw_args)
        self.index_count = defaultdict(set)
        self.signature_map = {}
        self.write_lock = Lock()

    def get_signatures(self, box_ids):
        """Extracts signatures for given box_ids
           This function uses self.signature_map as a caching mechanism
           to avoid making DB calls for getting signatures multiple times
           Args:
                box_ids: idx of boxes
           Returns:
                np array of features for given boxes
        """
        boxes_without_sigs = {b for b in box_ids if b not in self.signature_map}
        if boxes_without_sigs:
            with self.write_lock:
                boxes_without_sigs = {b for b in box_ids if b not in self.signature_map}
                if boxes_without_sigs:
                    new_feats = self.sig_extractor.extract(boxes_without_sigs)
                    self.signature_map.update({box_id:f for box_id,
                                f in zip(boxes_without_sigs, new_feats)})
        feats = [self.signature_map[b] for b in box_ids]
        return np.asarray(feats)

    def build_index(self, box_ids, label_ids):
        '''builds/updates the index given box and label ids'''
        boxes_info = []
        chunk_size = 1000
        t1 = datetime.utcnow()
        for i in range(0, len(box_ids), chunk_size):
            boxes_iter = box_ids[i:i+chunk_size]
            labels_iter = label_ids[i:i+chunk_size]
            feats = self.get_signatures(boxes_iter)
            boxes_info = [BoxInfo(box_id, feature, label_id) for box_id, feature, label_id in zip(boxes_iter, feats, labels_iter)]
            self.lsh.index(boxes_info)

        for b, l in zip(box_ids, label_ids):
            self.index_count[l].add(b)

        logger.info('Finished indexing for %d faces in %s ' %(len(box_ids), str(datetime.utcnow()-t1)))

    def query(self, feature):
        return self.lsh.query(feature)

    def get_neighbors(self, box, return_distances=True, num_neighbors='all', return_sorted=True):
        ''' method returns the sorted list of nearest neighbors for a given test boxid
            box : can be either a box_id or 1937 dim signature for a box
            returns --> sorted list of (box_id, label_id, distance) where len of list = num_neighbors
        '''
        neighbors = []
        if isinstance(box, (int, long)):
            feature = self.get_signatures([box])
            res = self.query(feature)
        elif isinstance(box, (list, np.ndarray)):
            arr = np.asarray(box)
            assert arr.shape == (1, FEATURE_LENGTH) or arr.shape == (FEATURE_LENGTH,), "Got wrong dimensions for box signature"
            arr = np.reshape(arr, (1, FEATURE_LENGTH))
            feature = self.sig_extractor.project_signature(arr)
            res = self.query(feature)
        else:
            raise(ValueError, "box can be either int (box_id) or 1937 dim signature (list or numpy.ndarray)")

        if res:
            if return_distances:
                box_ids, label_ids = zip(*res)
                feats = self.get_signatures(box_ids)
                # compute distance with all possible near neighbors
                dists = cdist(feature, feats)

                for box_id, label_id, dist in zip(box_ids, label_ids, dists[0]):
                    neighbors.append((box_id, label_id, dist))

                if return_sorted:
                    neighbors = sorted(neighbors, key = lambda x: x[2])
                    assert num_neighbors=='all' or isinstance(num_neighbors, int), 'num_neighbors should be either "all" or must be an integer'
                    if num_neighbors != 'all':
                        neighbors = neighbors[:num_neighbors]
            else:
                for box_id, label_id in res:
                    neighbors.append((box_id, label_id))
        return neighbors

    def _get_likelihood(self, neighbors, max_dist):
        assert max_dist > 0, "max_dist has to be positive"
        weights = defaultdict(int)
        for n in neighbors:
            _, label, dist = n
            # vote is inversely proportional to the distance
            vote = 1 - float(dist)/max_dist
            weights[label] += vote

        total = sum(weights.values())
        for k in weights.keys():
            weights[k] /= total

        items = weights.items()
        items.sort(key=lambda x:x[1], reverse=True)

        return items

    def predict(self, neighbors, max_dist, prob_delta):
        likelihood = self._get_likelihood(neighbors, max_dist)
        res_label = None
        if len(likelihood) > 1:
            if likelihood[0][1] >= likelihood[1][1] + prob_delta:
                res_label = likelihood[0][0]
        elif len(likelihood) == 1:
            res_label = likelihood[0][0]
        return res_label, likelihood

    def get_verdict(self, neighbors, k=15, radius=20.0, prob_delta=0.2):
        """ Gives Verdict given potential near neighbors
            neighbors : list of potential neighbors i.e. list of tuples (box_id, labl_id, dist)
            k : k for KNN
            radius : max radius beyond which no neighbors will be considered
            prob_delta : the minimum difference between the first and the second to give a verdict
        """
        res_label = likelihood = None
        if neighbors:
            neighbors.sort(key=lambda x:x[2])
            box_id, label, min_dist = neighbors[0]
            filter_neighbors = filter(lambda x:x[2] < radius, neighbors[:k])
            if len(filter_neighbors) >= k/3:
                res_label, likelihood = self.predict(filter_neighbors, radius, prob_delta)

        return res_label, likelihood

    def _encode(self):
        """Convert FaceIndex object to a dictionary of built-in data types"""
        face_index_dict = self.__dict__.copy()
        face_index_dict['lsh'] = self.lsh.encode()
        face_index_dict['sig_extractor'] = self.sig_extractor.encode()
        face_index_dict['signature_map'] = {}
        face_index_dict['write_lock'] = None
        return face_index_dict

    @classmethod
    def _decode(cls, face_index_dict):
        """Convert dictionary of built-in data types back into a new FaceIndex object"""
        face_index_dict['sig_extractor'] = AbstractSigExtractor.decode(face_index_dict['sig_extractor'])
        face_index_dict['lsh'] = LSH.decode(face_index_dict['lsh'])
        face_index = FaceIndex.__new__(FaceIndex)
        face_index.__dict__ = face_index_dict
        return face_index

    @classmethod
    def load_from_file(cls, file_name):
        """Load dict from a pickle file and decode"""
        with open(file_name, 'rb') as f:
            face_index_dict = pickle.load(f)
        face_index = cls._decode(face_index_dict)
        face_index.signature_map = {}
        face_index.write_lock = Lock()
        return face_index

    @classmethod
    def save_to_file(cls, face_index, file_name):
        """Encode and save to a pickle file for future loads"""
        enc_face_index = cls._encode(face_index)
        with open(file_name, 'wb') as f:
            pickle.dump(enc_face_index, f)
