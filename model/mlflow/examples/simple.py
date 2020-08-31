import numpy as np
from logging import getLogger

from affine.detection.model import PCA, LibsvmClassifier
from affine.detection.model.cross_validation import std_cross_validation
from affine.detection.model.mlflow import Step, Flow, FutureFlowInput

logger = getLogger(name=__name__)


class MathOp(object):

    def addone(self, data):
        return data + 1

    def times2(self, data):
        return data * 2

    def times3(self, data):
        return data * 3

    def add(self, data1, data2):
        return data1 + data2

    def print_data(self, data):
        print "Result is : %s" % data


def main():
    add_one = Step('add one', MathOp(), 'addone')
    times_two = Step('times_two', MathOp(), 'times2')
    times_three = Step('times_three', MathOp(), 'times3')
    add = Step('Add', MathOp(), 'add')
    print_step = Step('print', MathOp(), 'print_data')

    # Split and Merge
    f = Flow("test")
    f.add_step(add_one)
    f.add_step(times_two)
    f.add_step(times_three)
    f.add_step(add)
    f.add_step(print_step)

    f.start_with(add_one, FutureFlowInput(f, 'ip_data'))
    f.connect(add_one, times_two, add_one.output)
    f.connect(add_one, times_three, add_one.output)
    f.connect([times_two, times_three], add,
              times_two.output, times_three.output)
    f.connect(add, print_step, add.output)

    f.enable_log()
    f.run_flow(ip_data=1)
    return f


def main2():
    data = np.random.rand(500, 30)
    labels = np.asarray([int(i > 0.5) for i in data[:, 0]])

    # Train and Test
    pca = Step("PCA", PCA(ndims=20), 'train_and_project')
    svm = Step("SVM", LibsvmClassifier(), 'train')

    f = Flow('test2')
    f.add_step(pca)
    f.add_step(svm)

    f.start_with(pca, FutureFlowInput(f, 'data'))
    f.connect(pca, svm, pca.output, labels)

    f.run_flow(data=data)


def main3():
    data = np.random.rand(500, 30)
    labels = np.asarray([int(i > 0.5) for i in data[:, 0]])

    # Train and Cross validation
    pca = Step("PCA", PCA(ndims=20), 'train_and_project')
    cross_validation = Step('cv', std_cross_validation, None)

    f = Flow('test3')
    f.add_step(pca)
    f.add_step(cross_validation)

    f.start_with(pca, FutureFlowInput(f, 'data'))
    f.connect(pca, cross_validation, LibsvmClassifier(),
              pca.output, labels, n_folds=3)

    f.run_flow(data=data)


if __name__ == '__main__':
    main()
    main2()
    main3()
