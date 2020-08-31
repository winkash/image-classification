import os
import cv2
import numpy as np
import scipy
from logging import getLogger

from affine.detection.model import KNNScikit

__all__ = ["RobustMatcher", "HomographyModel", "ransac"]

logger = getLogger('affine.detection.model.robust_matching')

MIN_INLIERS_TH = 10


class RobustMatcher(object):

    def __init__(self, min_points, min_matches, ransac_th=3,
                 accept_th=0.3, knn_thresh=0.8,
                 ransac_algorithm=cv2.RANSAC, ransac_max_iter=100,
                 ransac_prob=0.95, inlier_r=0.5,
                 debug=False):
        """ Creates a robustmatcher object
        Args:
            min_points:  minimum number of points to perform matching
                between ImageDesc objects
            min_matches: minimum number of points to use to compute homography
            ransac_th: threshold for ransac
            accept_th: threshold for scores in prediction
            knn_thresh: threshold for nearest neighbors
            ransac_algorithm: int, cv2.RANSAC (or 8) means use the opencv ransac
                Otherwise, use the ransac implemented in this framework,
                which takes more params (max_iter, prob and inlier_r)
            ransac_max_iter: int, max number of iterations that ransac can run
                over-writing the required number computed automatically
            ransac_prob: float, probability of success in finding the best
                robust estimation, if one exists
                (the higher, the more iterations ransac will try)
            inlier_r: float, approximate ratio of inliers expected in the data
                used (the lower, the more iterations ransac will try)
            debug: if debug is true, you can run draw_match function to
                visualize the match
        """
        self.min_points = min_points
        self.min_matches = min_matches
        self.ransac_th = ransac_th
        self.accept_th = accept_th
        self.knn_thresh = knn_thresh
        self.debug = debug
        self.ransac_algorithm = ransac_algorithm
        self.ransac_max_iter = ransac_max_iter
        self.ransac_prob = ransac_prob
        self.inlier_r = inlier_r

    def match(self, query, image_list):
        """
        Computes a score between the query ImageDesc and a list of ImageDesc
        using robust point matching
        Args:
            query:  a ImageDesc object
            image_list:  a list of ImageDesc objects
        """
        if self.debug:
            self.query_kpts = []
            self.imgs_kpts = []
            self.mask_list = []

        num_feat_query = query.desc_rows
        if num_feat_query < self.min_points:
            return [0] * len(image_list)

        img_scores = []
        nbrs = KNNScikit(2, weights='distance', algorithm='brute')
        nbrs.train(query.descriptors, xrange(num_feat_query))
        for num_img, img in enumerate(image_list):
            inliers = 0
            mask = []
            matches_q, matches_im = [], []
            if img.desc_rows > self.min_points:
                dist, ind = nbrs.get_neighbors(img.descriptors, 2)
                for idx, ([dnn1, dnn2], [idx1, _]) in enumerate(zip(dist, ind)):
                    if dnn1 < dnn2 * self.knn_thresh:
                        matches_q.append(idx1)
                        matches_im.append(idx)
                inliers, mask = self._score_matches(
                    query.keypoints[matches_q], img.keypoints[matches_im])
            if self.debug:
                self.query_kpts.append(query.keypoints[matches_q])
                self.imgs_kpts.append(img.keypoints[matches_im])
                self.mask_list.append(mask)
            img_scores.append(float(inliers) / (len(matches_im) or 1))
            logger.debug('%d inliers out of %d features in query %d features in model-img' %
                        (inliers, num_feat_query, img.desc_rows))

        logger.debug('Scores obtained in robust matching %s' % str(img_scores))
        return img_scores

    def draw_matches(self, query_img, lists_img, final_scores, out_dir):
        """ Draws matches between the query image and all the candidate images
        and writes them in an image. The name of image would be final_scores list
        and it will be saved in out_dir
        Args:
            query_img: the full path to the query image
            lists_img: list of the full paths of candidate images
            final_scores: output scores of "match" function
            out_dir: directory to save the visualization image
        """
        assert self.debug is True, "You need to be in debug mode"
        for inst in ['query_kpts', 'imgs_kpts', 'mask_list']:
            assert hasattr(self, inst), "You need to run match function first"
        if len(self.query_kpts)==0:
            logger.debug(os.path.basename(query_img) + ' does not have enough keypoints')
            return
        img1 = cv2.imread(query_img)
        (rows1, cols1, _) = img1.shape
        candidate_images = [cv2.imread(img) for img in lists_img]
        candidate_coordinates = [
            img.shape if img is not None else (0, 0, 0) for img in candidate_images]
        blue = (255, 0, 0)
        candidates_max_column = max([a[1] for a in candidate_coordinates])
        candidates_sum_row = sum([a[0] for a in candidate_coordinates])
        maximum_rows = [max(rows1, cand_shape[0]) for cand_shape in candidate_coordinates]
        out_row = sum(maximum_rows)
        out_column = cols1 + candidates_max_column
        out = np.zeros((out_row, out_column, 3), dtype='uint8')
        out.fill(255)
        last_row = 0
        for idx, img2 in enumerate(candidate_images):
            if candidate_coordinates[idx] != (0, 0, 0):
                (rows2, cols2, _) = candidate_coordinates[idx]
                out[last_row:last_row + rows1, :cols1, :] = img1
                out[last_row:last_row + rows2, cols1:cols1 + cols2, :] = img2
                zipper = zip(self.query_kpts[idx], self.imgs_kpts[idx])
                if len(self.mask_list[idx]):
                    for pidx, (m1, m2) in enumerate(zipper):
                        if self.mask_list[idx][pidx] == 1:
                            (x1, y1) = (int(m1[0]), int(m1[1]))
                            (x2, y2) = (int(m2[0]), int(m2[1]))
                            cv2.circle(out, (x1, y1 + last_row), 4, blue, 3)
                            cv2.circle(
                                out, (x2 + cols1, y2 + last_row), 4, blue, 3)
                            cv2.line(
                                out, (x1, y1 + last_row), (x2 + cols1, y2 + last_row), blue, 1)
                last_row = max(rows1, rows2) + last_row
        final_scores_str = []
        for idx, cand in enumerate(candidate_coordinates):
            if cand!= (0, 0, 0):
                final_scores_str.append(str(round(final_scores[idx], 4)))
        scores_names = ''
        for element in final_scores_str:
            scores_names += element + '_'
        file_name = scores_names + os.path.basename(query_img)
        result_name = os.path.join(out_dir, file_name)
        cv2.imwrite(result_name, out)

    def _score_matches(self, q_points, im_points):
        """ Computes a score between a set of points finding an homography
            Args:
                q_points: query point matrix nx2, where each row is a cartesian
                point (x,y)
                im_points: a matrix of points to match
            Returns:
                score: a proportion of the points matched
            Raises:
                AssertionError: if the matrices have different sizes
        """
        assert(q_points.shape == im_points.shape), "different number of "\
            "points to match"
        mask = []
        num_inliers = 0
        if len(q_points) > self.min_matches:
            retval, mask = self._robust_h_estimation(q_points, im_points,
                                                     opt=self.ransac_algorithm)
            num_inliers = np.sum(mask)
        return num_inliers, mask

    def predict(self, scores, labels):
        """ Predicts the label for a given match
        Args:
            scores: a list of float numbers
            labels: a list of strings
        Returns:
            img: the label of the highest score, given a threshold.
            If threshold is higher, -1 is returned
        """
        assert len(scores) == len(labels),\
            "Labels and scores have different size"
        img = -1
        scores = [0 if scr>1 else scr for scr in scores]
        if np.max(scores) >= self.accept_th:
            img = labels[np.argmax(scores)]
            logger.info('Label %s found!, score %.2f' %
                        (str(img), np.max(scores)))
        return img

    def _robust_h_estimation(self, q_points, im_points, opt=cv2.RANSAC):
        """
            Robust estimation of an Homography and the set of inliers from the
            given set of matched points.

            Args:
                q_points: list of tuples (x,y) corresponding with points in
                    query image
                im_points: list of tuples (x,y) corresponding with points
                    matched in the reference image.
                    NOTE that this list should be aligned with q_points in such
                    a way that q_points[0] and im_points[0] are matched points
                opt: int, type of robust estimation.
                    If opt=cv2.RANSAC or opt=8 , this function uses the opencv
                        robust estimation.
                    Else this function uses the flexible ransac estimation
                        (including params prob and outlier_r)
        """
        if opt == cv2.RANSAC:
            homography, inlier_mask = cv2.findHomography(
                q_points, im_points, cv2.RANSAC, self.ransac_th)
        else:
            all_data = np.append(q_points, im_points, axis=1)
            homography, inlier_mask, best_err = ransac(
                data=all_data, model=HomographyModel,
                min_model_points=HomographyModel.MIN_POINTS_FOR_H,
                max_iter=self.ransac_max_iter, th=self.ransac_th,
                p=self.ransac_prob, w=self.inlier_r)
            inlier_mask = inlier_mask.reshape((len(inlier_mask), 1))
        return homography, inlier_mask


class HomographyModel(object):
    """
    Class for Homography estimation.
    It implements the interface needed by the ransac() function.
    """

    # An homography needs to have at least 4 points to be estimated
    MIN_POINTS_FOR_H = 4

    @staticmethod
    def fit(data):
        """
        Find and return perspective transformation H between the source (src)
        and the destination (dst) planes.

        The transformation minimizes the error of this constraint
            H * src_points = dst_points
        for all given points, using homogeneous coordinates
        i.e., it fits the homography to ALL given input data.

        Args:
            data: np array, each row: x_src, y_src, x_dst, y_dst
        Returns:
            H: 3x3 np array with the Homography parameters
        """
        q_points = np.array(data[:, :2])
        im_points = np.array(data[:, 2:])
        H, mask = cv2.findHomography(srcPoints=q_points, dstPoints=im_points,
                                     method=0)
        return H

    @staticmethod
    def get_error(data, model):
        """
        Compute the error when projecting source points into corresponding
        target points using the input model (Homography)

        Args:
            data: np array, each row: x_src, y_src, x_dst, y_dst
            model: homography matrix H, 3x3 np array, such that:
                        H * src_points = dst_points
                   using homogeneous coordinates
        Return
            list of reprojection error for each data point
        """
        list_ones = np.ones((data.shape[0], 1))
        src_p = np.append(data[:, :2], list_ones, axis=1)
        dst_p = np.append(data[:, 2:], list_ones, axis=1)
        src_p_transformed = scipy.dot(model, src_p.T)
        for i in range(3):
            # adding epsilon for non zero division
            src_p_transformed[i] /= (src_p_transformed[2] + 0.0000000001)
        err_per_point = scipy.sqrt(np.sum((dst_p.T - src_p_transformed) ** 2,
                                   axis=0))

        return err_per_point


def _assert_is_method(obj, method_name):
    """ check if an object has certain method available """
    return hasattr(obj, method_name) and callable(getattr(obj, method_name))


def ransac(data, model, min_model_points, max_iter, th, p=0.95, w=0.5):
    """
    Fit model to input data using the RANSAC algorithm

    Args:
        data: a set of observed data points (each row, one observation)
        model: model that can be fitted to data points.
            NOTE: This object/class should implement fit and get_error methods
        min_model_points: int,
            minimum number of data values required to fit the model
        max_iter: int, maximum number of iterations allowed in the algorithm
        th: float, threshold to determine when a data point is an inlier,
            (use the same units as the value returned by model.get_error)
            e.g., the reprojection error in pixels
        p: float [0,1], desired confidence (the larger, the more iterations)
        w: float [0,1], expected inlier ratio (the smaller, the more iterations)

    Return:
        bestfit: model which best fit the data (None if no good model is found)
        inliers: list of indx from input data which correspond to inliers
        best_err: average reprojection error for all inliers with best model

    Assertions:
        asserts if the model given does not implement fit or get_error methods
    """
    assert _assert_is_method(model, 'fit') and \
        _assert_is_method(model, 'get_error'), \
        "Model should provide fit and get_error methods"

    ransac_num_it = scipy.log(1 - p) / scipy.log(1 - (w ** min_model_points))
    if max_iter > 0:
        ransac_num_it = np.min([max_iter, int(ransac_num_it)])

    best_fit = None
    best_err = np.inf
    best_inlier_idxs = []
    num_data_points = data.shape[0]
    for iterations in range(ransac_num_it):
        candidate_idxs, test_idxs = random_partition(min_model_points,
                                                     num_data_points)
        candidate_inliers = data[candidate_idxs]
        test_points = data[test_idxs]
        candidate_model = model.fit(candidate_inliers)
        test_err = np.array(model.get_error(test_points, candidate_model))
        inlier_idxs = test_idxs[np.argwhere(test_err < th).flatten()]
        additional_inliers = data[inlier_idxs, :]

        if len(additional_inliers) > len(best_inlier_idxs) - min_model_points:

            current_data = np.concatenate((candidate_inliers,
                                           additional_inliers))
            current_model = model.fit(current_data)
            current_errs = model.get_error(current_data, current_model)
            current_avg_error = np.mean(current_errs)

            best_fit = current_model
            best_err = current_avg_error
            best_inlier_idxs = np.concatenate((candidate_idxs, inlier_idxs))

        if len(best_inlier_idxs) == num_data_points:
            break

    logger.debug('iterations done %d out of %d ' %
                (iterations + 1, ransac_num_it) +
                'Best model found: %d inliers out of %d matches. ' %
                (len(best_inlier_idxs), num_data_points) +
                'Mean error: %.5f' % best_err + '(th %.3f)' % (th))

    inliers_mask = np.zeros(num_data_points, dtype=int)
    if best_err <= th and len(best_inlier_idxs) >= MIN_INLIERS_TH:
        inliers_mask[best_inlier_idxs] = 1
    else:
        best_fit = None

    return best_fit, inliers_mask, best_err


def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2
