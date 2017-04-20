__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
sys.path.insert(0, 'ResearchTools')

from imports.basic_modules import *
from imports.import_caffe import *
from imports.ResearchTools import *
from imports.libmodules import *


class DenseSoftmax(caffe.Layer):
    def setup(self, bottom, top):
        assert (len(bottom) == 2)

        self.top_names = ['loss']
        top[0].reshape(1)

    def reshape(self, bottom, top):
        self.score = bottom[0].data
        self.prob = Jsoftmax(self.score, axis=1)
        self.labels = bottom[1].data
        pass

    def forward(self, bottom, top):
        labels = bottom[1].data

        loss = compute_dense_softmax_loss(self.prob, labels)
        top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        labels = bottom[1].data
        if propagate_down[0]:
            backprop = compute_dense_softmax_loss_grad(self.prob, labels)
            bottom[0].diff[...] = backprop
        pass


def compute_dense_softmax_loss(prob, label):
    """
    :param preds: dense prediction, Nx21x41x41
    :param labels: gt dense label, Nx1x41x41
    :return loss: scalar loss
    """

    N, C, H, W = prob.shape

    assert (H == label.shape[2] and W == label.shape[3])

    loss = 0
    batch_weight = 0

    for i in range(N):
        for jx in range(H):
            for jy in range(W):
                gt_label = label[i, 0, jx, jy].astype(np.int)
                if gt_label == 255:
                    continue
                elif 0 <= gt_label < C:
                    batch_weight += 1
                    loss -= np.log(max(prob[i, gt_label, jx, jy], 1e-10))
                else:
                    raise Exception('Unexpected label')

    return loss / batch_weight


def compute_dense_softmax_loss_grad(prob, label):
    N, C, H, W = prob.shape

    assert (H == label.shape[2] and W == label.shape[3])

    grad = np.zeros((N, C, H, W), dtype=np.float32)
    batch_weight = 0

    for i in range(N):
        for jx in range(H):
            for jy in range(W):
                gt_label = label[i, 0, jx, jy].astype(np.int)
                if gt_label == 255:
                    continue
                elif 0 <= gt_label < C:
                    batch_weight += 1
                    for c in range(C):
                        grad[i, c, jx, jy] = prob[i, c, jx, jy]
                    grad[i, gt_label, jx, jy] -= 1
                else:
                    raise Exception('Unexpected label')

    return grad / batch_weight
