import glob
import os
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from affine.detection.vision.facerec import *


def get_plot(res, rad, prob, ylabel, xlabel, name_plot):
    fig, ax = plt.subplots()
    for p in xrange(0, res.shape[0]):
        ax.plot(prob, res[p, :], 'o-', label='%.1f' % rad[p])
    ax.legend(ncol=len(prob) / 2, loc=0, prop={'size': 8})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.savefig('%s.pdf' % name_plot)


def mean_results(res):
    counter = 0
    meanres = np.zeros(res[0].shape)
    for r in res:
        meanres += r
        counter += 1
    meanres = np.divide(meanres, counter)
    return meanres


def aggregate(folder):
    myfiles = glob.glob(os.path.join(folder, '*.npz'))
    sel = []
    recall = []
    tpr = []
    fpr = []
    radius = []
    prob = []
    for f in myfiles:
        with open(f, 'r') as fo:
            data = np.load(fo)
            sel.append(data['arr_2'])
            recall.append(data['arr_3'])
            tpr.append(data['arr_0'])
            fpr.append(data['arr_1'])
            radius = data['arr_4']
            prob = data['arr_5']
    mean_tpr = mean_results(tpr)
    mean_fpr = mean_results(fpr)
    sel = np.array(sel).reshape((1, len(sel)))
    recall = np.array(recall).reshape((1, len(recall)))
    get_plot(mean_tpr, radius, prob, 'True Positive Rate',
             'Probability Threshold', 'tpr')
    get_plot(mean_tpr, radius, prob, 'True Positive Rate',
             'Probability Threshold', 'tpr')
    get_plot(sel, range(sel.shape[1]), range(
        sel.shape[1]), 'Selectivity', 'Experiment', 'sel')
    get_plot(recall, range(recall.shape[1]), range(
        recall.shape[1]), 'Recall', 'Experiment', 'recall')

    return mean_tpr, mean_fpr, recall, sel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', '--folder')
    args = parser.parse_args()
    aggregate(args.folder)
