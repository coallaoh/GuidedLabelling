__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
sys.path.insert(0, 'ResearchTools')

from imports.basic_modules import *
from imports.import_caffe import *
from imports.ResearchTools import *
from imports.libmodules import *

from image.crop import random_crop


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


def preprocess_convnet_image(im, transformer, input_size, phase, return_deprocess_confs=False):
    if phase == 'train':
        im = random_crop(im)
    elif phase == 'test':
        pass
    else:
        raise NotImplementedError

    imshape_postcrop = im.shape[:2]
    im = scipy.misc.imresize(im, input_size / float(max(imshape_postcrop)))

    imshape = im.shape[:2]
    margin = [(input_size - imshape[0]) // 2, (input_size - imshape[1]) // 2]
    im = cv2.copyMakeBorder(im, margin[0], input_size - imshape[0] - margin[0],
                            margin[1], input_size - imshape[1] - margin[1],
                            cv2.BORDER_REFLECT_101)
    assert (im.shape[0] == im.shape[1] == input_size)

    if phase == 'train':
        flip = np.random.choice(2) * 2 - 1
        im = im[:, ::flip, :]

    im = transformer.preprocess('data', im)
    if return_deprocess_confs:
        return im, dict(margin=margin, input_size=input_size, imshape=imshape, im_original_shape=imshape_postcrop)
    else:
        return im


def deprocess_convnet_label(label, confs):
    label_outshape = label.shape[1]
    label = nd.zoom(label,
                    [1, float(confs['input_size']) / label_outshape, float(confs['input_size']) / label_outshape],
                    order=1)

    label = label[:, confs['margin'][0]: (confs['imshape'][0] + confs['margin'][0]),
            confs['margin'][1]:(confs['imshape'][1] + confs['margin'][1])]

    label = nd.zoom(label, [1, float(confs['im_original_shape'][0]) / confs['imshape'][0],
                            float(confs['im_original_shape'][1]) / confs['imshape'][1]], order=1)

    return label


def preprocess_convnet_image_label(im, label, transformer, input_size, phase):
    if phase == 'train':
        im, coords = random_crop(im, return_coords=True)
        label = label[coords['x0']:coords['x1'], coords['y0']:coords['y1']]
    elif phase == 'test':
        pass
    else:
        raise NotImplementedError

    imshape_postcrop = im.shape[:2]
    im = scipy.misc.imresize(im, input_size / float(max(imshape_postcrop)))
    label = scipy.misc.imresize(label, input_size / float(max(imshape_postcrop)), interp='nearest', mode='F')

    imshape = im.shape[:2]
    margin = [(input_size - imshape[0]) // 2, (input_size - imshape[1]) // 2]
    im = cv2.copyMakeBorder(im, margin[0], input_size - imshape[0] - margin[0],
                            margin[1], input_size - imshape[1] - margin[1],
                            cv2.BORDER_REFLECT_101)
    label = cv2.copyMakeBorder(label, margin[0], input_size - imshape[0] - margin[0],
                               margin[1], input_size - imshape[1] - margin[1],
                               cv2.BORDER_REFLECT_101)
    assert (im.shape[0] == im.shape[1] == input_size)

    if phase == 'train':
        flip = np.random.choice(2) * 2 - 1
        im = im[:, ::flip, :]
        label = label[:, ::flip]

    im = transformer.preprocess('data', im)
    return im, label
