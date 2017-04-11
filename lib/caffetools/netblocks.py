__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
sys.path.insert(0, 'ResearchTools')

from imports.import_caffe import *


def get_frozen_param():
    return [dict(lr_mult=0)] * 2


def get_learned_param():
    weight_param = dict(lr_mult=1, decay_mult=1)
    bias_param = dict(lr_mult=2, decay_mult=0)
    return [weight_param, bias_param]


def get_frozen_param_single():
    return dict(lr_mult=0)


def get_learned_param_single():
    weight_param = dict(lr_mult=1, decay_mult=1)
    return weight_param


def get_fixed_param():
    weight_param = dict(lr_mult=0, decay_mult=0)
    bias_param = dict(lr_mult=0, decay_mult=0)
    return [weight_param, bias_param]


def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1, dilation=1,
              param=get_learned_param(),
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1), engine=0, in_place=False):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, dilation=dilation,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler, engine=engine)
    return conv, L.ReLU(conv, in_place=in_place)


def conv(bottom, ks, nout, stride=1, pad=0, group=1, dilation=1,
         param=get_learned_param(),
         weight_filler=dict(type='gaussian', std=0.01),
         bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, dilation=dilation,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv


def fc_relu(bottom, nout, param=get_learned_param(),
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1), in_place=False):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=in_place)


def fc(bottom, nout, param=get_learned_param(),
       weight_filler=dict(type='gaussian', std=0.005),
       bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc


def max_pool(bottom, ks, stride=1, pad=0):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride, pad=pad)


def scale(bottom, scale=1):
    return L.Power(bottom, power=1, scale=scale, shift=0)


def ptwisemult(bottom0, bottom1):
    return L.Eltwise(bottom0, bottom1, operation=0, stable_prod_grad=True)


def upsample(bottom, factor=1, channel=1):
    return L.Deconvolution(bottom,
                           convolution_param=dict(
                               weight_filler=dict(type='bilinear'),
                               kernel_size=2 * factor - factor % 2,
                               stride=factor,
                               num_output=channel,
                               group=channel,
                               pad=int(np.ceil((factor - 1.) / 2.)),
                               bias_term=False,
                           ),
                           param=dict(lr_mult=0, decay_mult=0)
                           )


def bw2rgb(bottom):
    return L.Concat(bottom, bottom, bottom, axis=1)
