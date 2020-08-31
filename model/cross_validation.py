from copy import deepcopy
from collections import defaultdict
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score


def std_cross_validation(clf, X, Y, n_folds=5, average='weighted', shuffle=False):
    """
    This function computes cross validation results with n_folds.
    The n folds are made preserving the percentage of samples for each class.

    Args:
        clf: classifier object (e.g. LibsvmClassifier,
            but any classifier that has similar train and test methods is valid)
        X: descriptors (each row contains the descriptor of one sample)
        Y: ground truth labels

        n_folds: int (default 5) number of folds for the cross validation
        shuffle: boolean (default False), whether to shuffle each stratification
            of the data before splitting into batches
        average: 'weighted' (default): Calculate metrics for each class,
                        and find their average, weighted by number of instances for each class.
                 'None': do not make any average, return metric for each label separately
    NOTE: If average is not None and the classification target is binary
        (only two classes given in Y), only class 1 scores will be returned.
        These options are the same as in sklearn.metrics

    Returns:
        dict with metrics from each iteration (fold) for each class
            metrics['precision'] (precision only for class 1 by default for binary classification,
                                list of precision for each label if average = None)
            metrics['recall'] (recall only for class 1 by default for binary classification,
                                list of precision for each label if average = None)
            metrics['accuracy'] (percentage of tests that were correct)

    """
    skf = StratifiedKFold(Y, n_folds, shuffle)
    metrics = defaultdict(list)
    for train_index, test_index in skf:
        test_clf = deepcopy(clf)
        test_clf.train(X[train_index], Y[train_index])
        Y_pred = test_clf.test(X[test_index])
        metrics['precision'].append(
            precision_score(Y[test_index], Y_pred, average=average))
        metrics['recall'].append(
            recall_score(Y[test_index], Y_pred, average=average))
        metrics['accuracy'].append(accuracy_score(Y[test_index], Y_pred))

    for m in metrics:
        metrics[m] = np.array(metrics[m])
        print '%s %s\nMean = %0.2f' % (m, metrics[m], np.mean(metrics[m]))

    return metrics
