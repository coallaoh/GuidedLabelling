__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
sys.path.insert(0, 'ResearchTools')

from imports.import_caffe import *
from caffetools.netblocks import get_learned_param, get_frozen_param, conv_relu, max_pool, conv

param = (get_frozen_param(), get_learned_param())


def vgg_conv1s(n, bottom, learn=1):
    n.conv1_1, n.relu1_1 = conv_relu(bottom, 3, 64, pad=1, param=param[learn])
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 3, 64, pad=1, param=param[learn])
    return n.relu1_2


def vgg_conv2s(n, bottom, learn=1):
    n.conv2_1, n.relu2_1 = conv_relu(bottom, 3, 128, pad=1, param=param[learn])
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 3, 128, pad=1, param=param[learn])
    return n.relu2_2


def vgg_conv3s(n, bottom, learn=1):
    n.conv3_1, n.relu3_1 = conv_relu(bottom, 3, 256, pad=1, param=param[learn])
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 3, 256, pad=1, param=param[learn])
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 3, 256, pad=1, param=param[learn])
    return n.relu3_3


def vgg_conv4s(n, bottom, learn=1):
    n.conv4_1, n.relu4_1 = conv_relu(bottom, 3, 512, pad=1, param=param[learn])
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 3, 512, pad=1, param=param[learn])
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 3, 512, pad=1, param=param[learn])
    return n.relu4_3


def deeplab_conv5s(n, bottom, learn=1):
    n.conv5_1, n.relu5_1 = conv_relu(bottom, 3, 512, pad=2, dilation=2, param=param[learn])
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 3, 512, pad=2, dilation=2, param=param[learn])
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 3, 512, pad=2, dilation=2, param=param[learn])
    return n.relu5_3


def deeplab_fc6(n, bottom, learn=1):
    n.fc6, n.relu6 = conv_relu(bottom, 3, 1024, pad=12, dilation=12, param=param[learn])
    n.drop6 = L.Dropout(n.relu6)
    return n.drop6


def deeplab_fc7(n, bottom, learn=1):
    n.fc7, n.relu7 = conv_relu(bottom, 1, 1024, param=param[learn])
    n.drop7 = L.Dropout(n.relu7)
    return n.drop7


def deeplab_fc8(n, bottom, learn=1):
    n.fc8_voc12 = L.Convolution(bottom, kernel_size=1, num_output=21,
                                param=[dict(lr_mult=10 * learn, decay_mult=1 * learn),
                                       dict(lr_mult=20 * learn, decay_mult=0 * learn)],
                                weight_filler=dict(type='gaussian', std=0.01),
                                bias_filler=dict(type='constant', value=0)
                                )
    return n.fc8_voc12


def deeplab_layers(n):
    conv1 = vgg_conv1s(n, n.data)
    n.pool1 = max_pool(conv1, ks=3, stride=2, pad=1)

    conv2 = vgg_conv2s(n, n.pool1)
    n.pool2 = max_pool(conv2, ks=3, stride=2, pad=1)

    conv3 = vgg_conv3s(n, n.pool2)
    n.pool3 = max_pool(conv3, ks=3, stride=2, pad=1)

    conv4 = vgg_conv4s(n, n.pool3)
    n.pool4 = max_pool(conv4, ks=3, stride=1, pad=1)

    conv5 = deeplab_conv5s(n, n.pool4)
    n.pool5 = max_pool(conv5, ks=3, stride=1, pad=1)

    fc6 = deeplab_fc6(n, n.pool5)
    fc7 = deeplab_fc7(n, fc6)
    scoremap = deeplab_fc8(n, fc7)

    return scoremap


def densesoftmaxloss(n, scoremap, label):
    n.loss = L.Python(
        scoremap,
        label,
        module='caffetools.losslayers', layer='DenseSoftmax', ntop=1, param_str=str(dict(
        )),
        loss_weight=1,
    )
    return n.loss


def deeplab(conf, control, phase):
    # setup the python data layer
    n = caffe.NetSpec()

    if phase == 'train':
        n.data, n.label = L.Python(module='caffetools.datalayers', layer='DeepLabData', ntop=2, param_str=str(dict(
            control=control,
            conf=conf,
        )))
    elif phase == 'test':
        n.data = L.DummyData(num=1, channels=3, height=conf['input_size'], width=conf['input_size'])
    else:
        raise NotImplementedError

    scoremap = deeplab_layers(n)

    if phase == 'train':
        loss = densesoftmaxloss(n, scoremap, n.label)

    return str(n.to_proto())
