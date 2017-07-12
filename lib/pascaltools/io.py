__author__ = 'joon'

import numpy as np
import os.path as osp
import random
import scipy
from xml.dom import minidom


def load_pascal_conf(control, conf):
    conf['nclasses'] = 20
    conf['year'] = '20' + control['test_dataset'][3:5]
    conf['testset'] = control['test_dataset'][5:]
    conf['imgsetpath'] = osp.join(conf['pascalroot'], 'VOC' + conf['year'], 'ImageSets', 'Segmentation', 'list',
                                  '%s.txt')
    if '_aug' in conf['testset']:
        conf['clsimgpath'] = osp.join(conf['pascalroot'], 'VOC' + conf['year'], 'SegmentationClassAug', '%s.png')
    else:
        conf['clsimgpath'] = osp.join(conf['pascalroot'], 'VOC' + conf['year'], 'SegmentationClass', '%s.png')

    conf['clsgt'] = osp.join(conf['pascalroot'], 'VOC' + conf['year'], 'ImageSets', 'Main',
                             '%s_' + conf['testset'] + '.txt')

    conf['imgpath'] = osp.join(conf['pascalroot'], 'VOC' + conf['year'], 'JPEGImages', '%s.jpg')

    return


def get_pascal_indexlist(root, year, type, split, shuffle=False, n=0, N=1, seed=132):
    pascal_list_file = osp.join(root, 'VOC' + year, 'ImageSets', type,
                                'list', split + '.txt')
    pascal_tmp_list = [line.rstrip('\n') for line in open(pascal_list_file)]
    if len(pascal_tmp_list[0].split()) == 1:
        indexlist = pascal_tmp_list
    else:
        indexlist = [line.split()[0].split('/')[2].split('.')[0] for line in pascal_tmp_list]
    indexlist = np.array(indexlist)

    if shuffle:
        random.seed(seed)
        random.shuffle(indexlist)

    if N > 1:
        start_idx = np.round(n * len(indexlist) / float(N))
        end_idx = np.round((n + 1) * len(indexlist) / float(N))
        indexlist = indexlist[start_idx:end_idx]

    return indexlist


def load_pascal_annotation(index, root, year):
    classes = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(classes, xrange(21)))

    filename = osp.join(root, 'VOC' + year, 'Annotations', index + '.xml')

    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
        data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, 21), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        x1 = float(get_data_from_tag(obj, 'xmin')) - 1
        y1 = float(get_data_from_tag(obj, 'ymin')) - 1
        x2 = float(get_data_from_tag(obj, 'xmax')) - 1
        y2 = float(get_data_from_tag(obj, 'ymax')) - 1
        cls = class_to_ind[
            str(get_data_from_tag(obj, "name")).lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'index': index}
