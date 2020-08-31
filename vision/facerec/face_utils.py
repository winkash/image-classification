import itertools
import cv, cv2
import numpy as np

'''
def tan_triggs_preprocessing(X, alpha=0.1, tau=10.0, gamma=0.2, sigma0=1.0, sigma1=2.0):
    X = np.array(X, dtype=np.float32)
    X = np.power(X, gamma)
    X = np.asarray(ndimage.gaussian_filter(X, sigma1) - ndimage.gaussian_filter(X, sigma0))
    X = X / np.power(np.mean(np.power(np.abs(X), alpha)), 1.0/alpha)
    X = X / np.power(np.mean(np.power(np.minimum(np.abs(X), tau), alpha)), 1.0/alpha)
    X = tau*np.tanh(X/tau)
    res = np.array(X)
    res = cv2.normalize(X, res, 0, 255, cv2.NORM_MINMAX, cv.CV_8UC1)
    return res
'''

def list2mat(ip_list, mat):
    assert mat.rows==len(ip_list) and mat.cols==len(ip_list[0]), "dimensions do not match, given list(%s, %s) and mat(%s, %s)" %(len(ip_list), len(ip_list[0]), mat.rows, mat.cols)
    for r, row in enumerate(ip_list):
        for c, val in enumerate(row):
            mat[r, c] = val
        
    return mat


def get_affine_transformation(src_points, dst_points, indices):
    '''get affine transformtaion for list of points'''
    src_pts = [s for i, s in enumerate(src_points) if i in indices]
    dst_pts = [s for i, s in enumerate(dst_points) if i in indices]

    srcm = cv.CreateMat(3,2, cv.CV_32FC1)
    dstm = cv.CreateMat(3,2, cv.CV_32FC1)
    src = np.asarray(list2mat(src_pts, srcm))
    dst = np.asarray(list2mat(dst_pts, dstm))
    H = cv2.getAffineTransform(src, dst)
    
    # rotation and translation
    R = H[:,:2]
    T = H[:, 2]
    
    err = 0.0
    #projection error calculation
    for spt, dpt in zip( np.asarray(src_points), np.asarray(dst_points)):
        actual = np.dot(R, spt) + T
        err += np.linalg.norm(dpt-actual)
    
    return H, err 


def get_best_fit(src_points, dst_points):
    '''performs mutiple iterations to get affine transformation with least error'''
    emin = 100000
    for indices in itertools.combinations(range(len(src_points)), 3):
        H, err = get_affine_transformation(src_points, dst_points, indices)
        if err < emin:
            bestH = H
            emin = err
    
    return bestH, emin
