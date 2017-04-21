#!/usr/bin/env python

__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
sys.path.insert(0, 'ResearchTools')

from imports.basic_modules import *
from imports.import_caffe import *
from imports.ResearchTools import *
from imports.libmodules import *

from config import config_test
from networks import deeplab

####

EXP_PHASE = 'seg-test'

conf = dict(
    save=True,
    vis=True,
    shuffle=True,
    overridecache=True,
    pascalroot="/BS/joon_projects/work/",
    imagenetmeanloc="data/ilsvrc_2012_mean.npy",
    gpu=1,
)

control = dict(
    init='VGG_ILSVRC_16_layers-deeplab',
    net='DeepLab',
    dataset='voc12train_aug',
    datatype='Segmentation',
    base_lr=0.001,
    batch_size=15,

    # seed
    s_g_init='VGG_ILSVRC_16_layers',
    s_g_net='GAP-HighRes',
    s_g_dataset='voc12train_aug',
    s_g_datatype='Segmentation',
    s_g_base_lr=0.001,
    s_g_batch_size=15,
    s_g_balbatch='clsbal',
    s_g_test_iter=8000,
    s_g_test_dataset='voc12train_aug',
    s_g_test_datatype='Segmentation',
    s_g_test_ranking='none',
    s_g_test_gtcls='use',

    # SAL
    s_s_net='DeepLabv2_ResNet',
    s_s_dataset='MSRA',
    s_s_datatype='NP',
    s_s_test_dataset='voc12train_aug',
    s_s_test_datatype='Segmentation',

    s_gtcls='use',
    s_seedthres=20,
    s_salthres=50,
    # s_guiderule='G0',
    # s_guiderule='G1',
    s_guiderule='G2',
    # s_guiderule='salor',
    s_test_dataset='voc12train_aug',
    s_test_datatype='Segmentation',

    test_iter=8000,
    test_dataset='voc12val',
    test_datatype='Segmentation',
    test_pcrf='deeplab',
)


####

def parse_input(argv=sys.argv):
    parser = argparse.ArgumentParser(description="Tests a segmentation network")
    parser.add_argument('--test_iter', default=8000, type=int,
                        help='Initialisation for the network')
    parser.add_argument('--test_dataset', default='voc12val', type=str,
                        help='Test set')
    parser.add_argument('--test_datatype', default='Segmentation', type=str,
                        help='Type of test set')

    parser.add_argument('--init', default='VGG_ILSVRC_16_layers-deeplab', type=str,
                        help='Initialisation for the network')
    parser.add_argument('--net', default='DeepLab', type=str,
                        help='Network')
    parser.add_argument('--dataset', default='voc12train_aug', type=str,
                        help='Training set')
    parser.add_argument('--datatype', default='Segmentation', type=str,
                        help='Type of training set')
    parser.add_argument('--base_lr', default=0.001, type=float,
                        help='Base learning rate')
    parser.add_argument('--batch_size', default=15, type=int,
                        help='Batch size')

    parser.add_argument('--s_g_init', default='VGG_ILSVRC_16_layers', type=str,
                        help='Initialisation for the network')
    parser.add_argument('--s_g_net', default='GAP-HighRes', type=str,
                        help='Network')
    parser.add_argument('--s_g_dataset', default='voc12train_aug', type=str,
                        help='Training set')
    parser.add_argument('--s_g_datatype', default='Segmentation', type=str,
                        help='Type of training set')
    parser.add_argument('--s_g_base_lr', default=0.001, type=float,
                        help='Base learning rate')
    parser.add_argument('--s_g_batch_size', default=15, type=int,
                        help='Batch size')
    parser.add_argument('--s_g_balbatch', default='clsbal', type=str,
                        help='Class balanced batch composition')
    parser.add_argument('--s_g_test_iter', default=8000, type=int,
                        help='Test model iteration')
    parser.add_argument('--s_g_test_dataset', default='voc12train_aug', type=str,
                        help='Test dataset')
    parser.add_argument('--s_g_test_datatype', default='Segmentation', type=str,
                        help='Type of test data')
    parser.add_argument('--s_g_test_ranking', default='none', type=str,
                        help='When testing, dont rank priority according to size @ 20 percent max score as in lamperts')
    parser.add_argument('--s_g_test_gtcls', default='use', type=str,
                        help='Use GT class information at test time')

    parser.add_argument('--s_s_net', default='DeepLabv2_ResNet', type=str,
                        help='Network')
    parser.add_argument('--s_s_dataset', default='MSRA', type=str,
                        help='Training set')
    parser.add_argument('--s_s_datatype', default='NP', type=str,
                        help='Type of training set')
    parser.add_argument('--s_s_test_dataset', default='voc12train_aug', type=str,
                        help='Test dataset')
    parser.add_argument('--s_s_test_datatype', default='Segmentation', type=str,
                        help='Type of test data')

    parser.add_argument('--s_gtcls', default='use', type=str,
                        help='Use GT class information at test time')
    parser.add_argument('--s_seedthres', default=50, type=int,
                        help='FG threshold for seeds')
    parser.add_argument('--s_salthres', default=50, type=int,
                        help='FG threshold for saliency')
    parser.add_argument('--s_guiderule', default='G2', type=str,
                        help='Rule for generating guide labels')
    parser.add_argument('--s_test_dataset', default='voc12train_aug', type=str,
                        help='Test dataset')
    parser.add_argument('--s_test_datatype', default='Segmentation', type=str,
                        help='Type of test data')
    control = vars(parser.parse_known_args(argv)[0])

    parser_conf = argparse.ArgumentParser()
    parser_conf.add_argument('--pascalroot', default='/home', type=str,
                             help='Pascal VOC root folder')
    parser_conf.add_argument('--imagenetmeanloc', default='/home', type=str,
                             help='Imagenet mean image location')
    parser_conf.add_argument('--gpu', default=1, type=int,
                             help='GPU ID')
    parser_conf.add_argument('--vis', default=False, type=bool,
                             help='Visualisation')
    parser_conf.add_argument('--save', default=True, type=bool,
                             help='Save raw heatmap output')
    parser_conf.add_argument('--overridecache', default=True, type=bool,
                             help='Override cache')
    parser_conf.add_argument('--shuffle', default=True, type=bool,
                             help='Shuffle test input order')
    conf = vars(parser_conf.parse_known_args(argv)[0])
    return control, conf


def write_proto(testproto, conf, control):
    f = open(testproto, 'w')
    f.write(deeplab(conf, control, 'test'))
    f.close()
    return


def setup_net(control, control_model, control_token, conf):
    # prototxt
    protodir = osp.join('models', EXP_PHASE, create_token(control_token))
    mkdir_if_missing(protodir)
    testproto = osp.join(protodir, 'test.prototxt')
    write_proto(testproto, conf, control)

    learnedmodel_dir = osp.join('cache', 'seg-train', create_token(control_model))
    learnedmodel = osp.join(learnedmodel_dir, '_iter_' + str(control['test_iter']) + '.caffemodel')

    # init
    caffe.set_mode_gpu()
    caffe.set_device(conf['gpu'])

    net = caffe.Net(testproto, learnedmodel, caffe.TEST)
    disp_net(net)

    return net


def run_test(net, out_dir, control, conf):
    year = '20' + control['test_dataset'][3:5]
    pascal_list = get_pascal_indexlist(conf['pascalroot'], year, control['test_datatype'], control['test_dataset'][5:],
                                       shuffle=conf['shuffle'])

    num_test = len(pascal_list)
    print('%d images for testing' % num_test)

    transformer = set_preprocessor_without_net(
        [1, 3, conf['input_size'], conf['input_size']],
        mean_image=np.load(conf['imagenetmeanloc']).mean(1).mean(1))

    pred_list = []
    id_list = []

    start_time = time.time()
    for idx in range(num_test):
        end_time = time.time()
        print ('    Iter %d took %2.1f seconds' % (idx, end_time - start_time))
        start_time = time.time()
        print ('    Running %d out of %d images' % (idx + 1, num_test))

        inst = idx
        im_id = pascal_list[inst]
        outfile = osp.join(out_dir, im_id + '.png')

        if conf['save']:
            if not conf['overridecache']:
                if osp.isfile(outfile):
                    print('skipping')
                    continue

        gtfile = conf['clsimgpath'] % (im_id)
        gt = np.array(Image.open(gtfile)).astype(np.float)

        imloc = os.path.join(conf['pascalroot'], 'VOC' + year, 'JPEGImages', im_id + '.jpg')
        image = load_image_PIL(imloc)
        imshape_original = image.shape[:2]

        net.blobs['data'].data[...][0] = preprocess_convnet_image(image, transformer, 321, 'test')
        net.forward()

        scoremap = net.blobs['fc8_voc12'].data[0]
        scoremap_originalshape = nd.zoom(scoremap, [1, imshape_original[0] / float(conf['output_size']),
                                                    imshape_original[1] / float(conf['output_size'])],
                                         order=1).transpose((1, 2, 0))
        seg_raw = scoremap_originalshape.argmax(2)

        if control['test_pcrf'] != 'none':
            probmap_originalshape = Jsoftmax(scoremap_originalshape, axis=2)
            probmap_originalshape_smth = CRF(image.copy(), probmap_originalshape, crf_param=control['test_pcrf'])
            seg = probmap_originalshape_smth.argmax(2).astype(np.uint8)
        else:
            seg = seg_raw.copy().astype(np.uint8)

        if conf['vis']:

            def visualise_data():
                fig = plt.figure(0, figsize=(15, 10))
                fig.suptitle('ID:{}'.format(im_id))
                ax = fig.add_subplot(2, 3, 1)
                ax.set_title('Original image')
                ax.imshow(image)
                ax = fig.add_subplot(2, 3, 4)
                ax.imshow(image)
                ax.imshow(gt, alpha=.5, cmap="nipy_spectral", clim=(0, 30))

                ax = fig.add_subplot(2, 3, 2)
                ax.set_title('Raw output')
                ax.imshow(seg_raw, cmap="nipy_spectral", clim=(0, 30))
                ax = fig.add_subplot(2, 3, 5)
                ax.imshow(image)
                ax.imshow(seg_raw, alpha=.5, cmap="nipy_spectral", clim=(0, 30))

                ax = fig.add_subplot(2, 3, 3)
                ax.set_title('Postprocessed')
                ax.imshow(seg, cmap="nipy_spectral", clim=(0, 30))
                ax = fig.add_subplot(2, 3, 6)
                ax.imshow(image)
                ax.imshow(seg, alpha=.5, cmap="nipy_spectral", clim=(0, 30))

                for iii in range(6):
                    fig.axes[iii].get_xaxis().set_visible(False)
                    fig.axes[iii].get_yaxis().set_visible(False)

                plt.pause(1)
                return

            visualise_data()

        if conf['save']:
            if not conf['overridecache']:
                assert (not os.path.isfile(outfile))
            else:
                if os.path.isfile(outfile):
                    print('WARNING: OVERRIDING EXISTING RESULT FILE')
            cv2.imwrite(outfile, seg)
            print('results saved to %s' % outfile)
    return


def main(control, conf):
    control, control_model, control_token, conf = config_test(control, conf, EXP_PHASE)

    out_dir = osp.join('cache', EXP_PHASE, create_token(control_token))
    mkdir_if_missing(out_dir)
    print('saving to: {}'.format(out_dir))

    net = setup_net(control, control_model, control_token, conf)

    run_test(net, out_dir, control, conf)
    return


if __name__ == '__main__':
    if len(sys.argv) != 1:
        control, conf = parse_input(sys.argv)
    main(control, conf)
