__author__ = 'joon'

import sys

sys.path.insert(0, 'lib')

from imports.import_caffe import *


def set_preprocessor(net, mean_image=None):
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    if mean_image is not None:
        transformer.set_mean('data', mean_image)  # subtract the dataset-mean value in each channel
    # transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    return transformer


def set_preprocessor_without_net(data_shape, mean_image=None):
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': data_shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    if mean_image is not None:
        transformer.set_mean('data', mean_image)  # subtract the dataset-mean value in each channel
    # transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    return transformer


def deprocess_net_image(image, subtract_mean=True):
    image = image.copy()  # don't modify destructively
    if (image.max() - image.min()) > 300:
        image /= (image.max() - image.min())
    image = image[::-1]  # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    if subtract_mean:
        image += [123, 117, 104]  # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image
