"""
This utility module wraps the various BOVW binaries, making sure that
their required input files are readable and output files writable.
"""
import os
import cv2.cv as cv
import numpy as np

from os.path import join
from xml.etree import ElementTree

from affine.video_processing import run_cmd
from affine.detection.model.classifiers import LibsvmClassifier

def verify_files_readable(files):
    """Verifies files in list 'files' exist and are readable
    'files' can also be a single filename string.
    Raises an IOError if any of the files aren't readable.
    """
    if type(files) == type('string'):
        files = [files]
    for file in files:
        if not (os.path.isfile(file) and os.access(file, os.R_OK)):
            raise IOError('Failed trying to read file %s' % file)
        if os.stat(file)[6] == 0:
            raise IOError('Empty file %s' % file)

def verify_files_writable(files):
    "Verifies files in list 'files' are creatable or writable"
    if type(files) == type('string'):
        files = [files]
    for file in files:
        ok = True
        if os.path.isfile(file):
            if not os.access(file, os.W_OK):
                ok = False
        else:
            if not os.access(os.path.dirname(file), os.W_OK):
                ok = False
        if not ok:
            raise IOError('Failed writing to file %s' % file)

def feature_extract(bin_dir, info_filename, extract_filename, params):
    if params['in_feature_type'] == 'HOG':
        hog_feature_extract(bin_dir, info_filename, extract_filename,
                            params['hog_variant'], params['cell_size'],
                            params['num_orientations'])
    elif params['in_feature_type'] == 'SURF' or params['in_feature_type'] == 'SURFEX':
        dense_feature_extract(bin_dir, info_filename, extract_filename,
                              params['in_feature_type'], params['in_intv'],
                              params['in_nLayer'], params['in_size'],
                              params['in_angle'])
    elif params['in_feature_type'] == 'ColorSURFEX':
        color_surfex_feature_extract(bin_dir, info_filename, extract_filename,
                                     params['in_color_type'], params['in_intv'],
                                     params['in_nLayer'], params['in_size'],
                                     params['in_angle'])
    else:
        raise ValueError("Feature type %s not found" % params['in_feature_type'])

def dense_feature_extract(bin_dir, in_file, out_file, in_feature_type, in_intv,
                          in_nLayer, in_size, in_angle):
    """ Compute features using a dense grid """
    verify_files_readable(in_file)
    verify_files_writable(out_file)
    ### only feature types: SURFEX(surf extended) and SURF
    if in_feature_type == "SURFEX":
        run_cmd([join(bin_dir, 'dense_feature_extract'), in_file, out_file,
                 'SURF', str(in_intv), str(in_nLayer), str(in_size),
                 str(in_angle), str(1)])
    elif in_feature_type == "SURF": ##in_feature_type == "SURF"
        run_cmd([join(bin_dir, 'dense_feature_extract'), in_file, out_file,
                 'SURF', str(in_intv), str(in_nLayer), str(in_size),
                 str(in_angle), str(0)])
    else:
        raise IOError('Feature %s not supported' % in_feature_type)

def hog_feature_extract(bin_dir, in_file, out_file, hog_feature_type, cell_size,
                        numDimensions):
    """ Compute HOG features """
    verify_files_readable(in_file)
    verify_files_writable(out_file)
    command = [join(bin_dir, 'hog_feature_extract'), in_file, out_file,
               str(hog_feature_type), str(cell_size), str(numDimensions)]
    run_cmd(command)

def color_surfex_feature_extract(bin_dir, in_file, out_file, color_type, in_intv,
                                 in_nLayer, in_size, in_angle):
    """ Compute Color SURFEX """
    verify_files_readable(in_file)
    verify_files_writable(out_file)
    command = [join(bin_dir, 'color_surfex_feature_extract'), in_file, out_file,
               str(color_type), str(in_intv), str(in_nLayer), str(in_size),
               str(in_angle)]
    run_cmd(command)

def pca_computation(bin_dir, in_file, in_pca_dimension, out_file):
    verify_files_readable(in_file)
    verify_files_writable(out_file)
    if in_pca_dimension <= 0:
        raise IOError('pca dimension is not a positive integer \n')
    run_cmd([join(bin_dir, 'pca_computation'), in_file, out_file,
             str(in_pca_dimension)])

def pca_projection(bin_dir, in_file_extract, in_file_pca, out_file):
    verify_files_readable([in_file_extract, in_file_pca])
    verify_files_writable(out_file)
    run_cmd([join(bin_dir, 'pca_projection'), in_file_extract, in_file_pca,
             out_file ])

def vocab_kms(bin_dir, in_file_projection, in_vocab_size, out_file):
    verify_files_readable(in_file_projection)
    verify_files_writable(out_file)
    document = ElementTree.parse(in_file_projection)
    data = document.find( 'nDescriptors')
    if in_vocab_size > int(data.text):
        raise IOError('number of codes is larger than number of descriptors \n')
    run_cmd([join(bin_dir, 'vocab_kms'), in_file_projection, out_file,
             str(in_vocab_size)])


def hist_kms(bin_dir, in_file_vocab, in_file_projection, out_file):
    verify_files_readable([in_file_vocab, in_file_projection])
    verify_files_writable(out_file)
    run_cmd([
        join(bin_dir, 'hist_kms'),
        in_file_vocab,
        in_file_projection,
        out_file,
        ])


def svm_model(in_file_hist_agg, svm_type, svm_kernel, out_file):
    hist = np.asarray(cv.Load(in_file_hist_agg))
    # The first col is the labels for the feature points
    # The last two cols are (vieo_id, timestamps)
    labels = hist[:, 0]
    features = hist[:, 1:-2]
    clf = LibsvmClassifier(svm_type=int(svm_type), kernel_type=int(svm_kernel))
    clf.train(features, labels)
    clf.save_to_file(out_file)


def predict(feature_file, model_file):
    clf = LibsvmClassifier.load_from_file(model_file)
    # The first col is the labels for the feature points
    # In case of prediction, the first col should be ingnored
    # The last two cols are (vieo_id, timestamps)
    test_vecs = np.asarray(cv.Load(feature_file))
    test_feats = test_vecs[:, 1:-2]
    metadata = test_vecs[:, -2:]
    labels = clf.predict(test_feats)

    results = {}
    for row, l in zip(metadata, labels):
        video_id, timestamp = row
        results[(video_id, timestamp)] = l

    return results
