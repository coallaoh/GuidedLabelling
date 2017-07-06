__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
sys.path.insert(0, 'ResearchTools')

from imports.basic_modules import *
from imports.import_caffe import *
from imports.ResearchTools import *
from imports.libmodules import *


class GAPData(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['data', 'label']
        params = eval(self.param_str)

        self.control = params['control']
        self.conf = params['conf']
        self.batch_loader = BatchLoader_PASCAL(params)

        top[0].reshape(self.control['batch_size'], 3, self.conf['input_size'], self.conf['input_size'])
        top[1].reshape(self.control['batch_size'], self.conf['nposcls'])

    def forward(self, bottom, top):
        for itt in range(self.control['batch_size']):
            im, label = self.batch_loader.load_next_image()
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass


class BatchLoader_PASCAL(object):
    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params):
        self.control = params['control']
        self.conf = params['conf']

        self.pascal_year = '20' + self.control['dataset'][3:5]
        self.pascal_split = self.control['dataset'][5:]
        self.pascal_type = self.control['datatype']

        self.indexlist = get_pascal_indexlist(self.conf['pascalroot'], self.pascal_year, self.pascal_type,
                                              self.pascal_split)

        self.transformer = set_preprocessor_without_net(
            [self.control['batch_size'], 3, self.conf['input_size'], self.conf['input_size']],
            mean_image=np.load(self.conf['imagenetmeanloc']).mean(1).mean(1))

        print "BatchLoader initialized with {} images".format(len(self.indexlist))
        self.indexlist_original = self.indexlist.copy()

        self._cur = 0
        self.start = True

    def pascal_balancedbatch(self, indexlist, seed=543):
        """
        Computes a sequence of inputs for convnet such that each mini-batch is class-balanced.
        """
        np.random.seed(seed)
        print('computing class balanced batch...')

        labellist = []
        start_t = time.time()
        for idx, index in enumerate(indexlist):
            im_id = index
            end_t = time.time()
            if end_t - start_t > 10:
                print('loading gt for %d out of %d' % (idx, len(indexlist)))
                start_t = time.time()
            label = self.get_pascal_gt(im_id)
            labellist.append(label)

        # Get instances for each class
        imgIds_cls = [None] * 20
        for idx in range(1, 21):
            imgIds_cls[idx - 1] = np.where(np.array([idx in label for label in labellist]))[0]

        insts = []

        min_len = min([len(imids) for imids in imgIds_cls])
        while (min_len > 0):
            cls_list = range(1, 21)
            np.random.shuffle(cls_list)
            NO_MORE = False
            while (len(cls_list) > 0):
                NO_MORE = False
                pick = cls_list[0]
                list_inst_cls = imgIds_cls[pick - 1].copy()
                SEEN_CLS = True
                while (SEEN_CLS):
                    if len(list_inst_cls) == 0:
                        NO_MORE = True
                        break
                    SEEN_CLS = False
                    id_choice = np.random.choice(list_inst_cls, 1)[0]
                    all_cls = labellist[id_choice]
                    for cls in all_cls:
                        if cls not in cls_list:
                            SEEN_CLS = True
                            list_inst_cls = list_inst_cls[list_inst_cls != id_choice]
                            break
                if NO_MORE:
                    break
                for cls in all_cls:
                    if cls in cls_list:
                        cls_list.remove(cls)
                imgIds_cls[pick - 1] = imgIds_cls[pick - 1][imgIds_cls[pick - 1] != id_choice]
                insts.append(id_choice)
                if len(imgIds_cls[pick - 1]) == 0:
                    break

            if NO_MORE:
                break
            min_len = min([len(imids) for imids in imgIds_cls])

        print('done')
        return np.array(indexlist)[np.hstack(insts)].tolist()

    def load_next_image(self):
        # Did we finish an epoch?
        if self._cur == len(self.indexlist) or self.start:
            self.start = False
            self._cur = 0
            if self.control['balbatch'] == 'none':
                np.random.shuffle(self.indexlist)
            elif self.control['balbatch'] == 'clsbal':
                seed = np.random.choice(range(10000), 1)
                self.indexlist = self.pascal_balancedbatch(self.indexlist_original, seed=seed)
            else:
                raise NotImplementedError

        # <<IMAGE>>
        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        image_file_name = osp.join(self.conf['pascalroot'], 'VOC' + self.pascal_year, 'JPEGImages', index + '.jpg')
        im = load_image_PIL(image_file_name)

        # Process
        im = preprocess_convnet_image(im, self.transformer, self.conf['input_size'], 'train')

        # <<LABEL>>
        multilabel = np.zeros(20).astype(np.float32)

        if '_aug' in self.pascal_split:
            gtpath = osp.join(self.conf['pascalroot'], 'VOC' + self.pascal_year, 'SegmentationClassAug', '%s.png')
        else:
            gtpath = osp.join(self.conf['pascalroot'], 'VOC' + self.pascal_year, 'SegmentationClass', '%s.png')

        gtfile = gtpath % (index)
        gt = np.array(Image.open(gtfile)).astype(np.float)
        gt_cls = np.unique(gt)
        gt_cls_nobg = gt_cls[(gt_cls != 0) & (gt_cls != 255)]
        for cls in gt_cls_nobg:  # anns['gt_classes'] is 0 for bg, 1~20 for fg categories
            multilabel[int(cls) - 1] = 1

        self._cur += 1
        return im, multilabel

    def get_pascal_gt(self, index):
        anns = load_pascal_annotation(index, self.conf['pascalroot'], self.pascal_year)
        return np.unique(anns['gt_classes'])


class DeepLabData(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['data', 'label']
        params = eval(self.param_str)

        self.control = params['control']
        self.conf = params['conf']
        self.batch_loader = BatchLoader_PASCAL_Labelling(params)

        top[0].reshape(self.control['batch_size'], 3, self.conf['input_size'], self.conf['input_size'])
        top[1].reshape(self.control['batch_size'], 1, self.conf['output_size'], self.conf['output_size'])

    def forward(self, bottom, top):
        for itt in range(self.control['batch_size']):
            im, label = self.batch_loader.load_next_image()
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass


class BatchLoader_PASCAL_Labelling(object):
    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params):
        self.control = params['control']
        self.conf = params['conf']

        self.pascal_year = '20' + self.control['dataset'][3:5]
        self.pascal_split = self.control['dataset'][5:]
        self.pascal_type = self.control['datatype']

        self.indexlist = get_pascal_indexlist(self.conf['pascalroot'], self.pascal_year, self.pascal_type,
                                              self.pascal_split)

        self.transformer = set_preprocessor_without_net(
            [self.control['batch_size'], 3, self.conf['input_size'], self.conf['input_size']],
            mean_image=np.load(self.conf['imagenetmeanloc']).mean(1).mean(1))

        print "BatchLoader initialized with {} images".format(len(self.indexlist))
        self.indexlist_original = self.indexlist.copy()

        self._cur = 0
        self.start = True

    def load_next_image(self):
        # Did we finish an epoch?
        if self._cur == len(self.indexlist) or self.start:
            self.start = False
            self._cur = 0
            np.random.shuffle(self.indexlist)

        # <<LOAD>>
        # Load image
        index = self.indexlist[self._cur]  # Get the image index
        image_file_name = osp.join(self.conf['pascalroot'], 'VOC' + self.pascal_year, 'JPEGImages', index + '.jpg')
        im = load_image_PIL(image_file_name)

        # Load label
        label_file_name = osp.join(self.conf['sup_dir'], index + '.png')
        label = load_image_PIL(label_file_name)

        # Process
        im, label = preprocess_convnet_image_label(im, label, self.transformer, self.conf['input_size'], 'train')
        label = scipy.misc.imresize(label, [self.conf['output_size'], self.conf['output_size']], interp='nearest',
                                    mode='F').astype(np.float32)

        self._cur += 1
        return im, label
