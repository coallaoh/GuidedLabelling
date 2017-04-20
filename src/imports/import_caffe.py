__author__ = 'joon'

caffe_root = 'caffe/'

import sys

sys.path.insert(0, caffe_root + 'python')
import caffe  # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

sys.path.append(caffe_root + "examples/pycaffe")  # the tools file is in this folder
import tools  # this contains some tools that we need

sys.path.insert(0, 'lib')
from caffetools.disp import disp_net
from caffetools.io import save_to_caffemodel
from caffetools.preprocessor import set_preprocessor_without_net, preprocess_convnet_image, \
    preprocess_convnet_image_label
from caffetools.debug import debug_get_diff, debug_get_data

from caffe import layers as L
from caffe import params as P
