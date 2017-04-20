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

from networks import deeplab
from config import config_train

####

EXP_PHASE = 'seg-train'

conf = dict(
    vis=False,
    save=True,
    overridecache=True,
    pascalroot="/BS/joon_projects/work/",
    imagenetmeanloc="data/ilsvrc_2012_mean.npy",
    gpu=2,
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
    s_seedthres=50,
    s_salthres=50,
    # s_guiderule='G0',
    # s_guiderule='G1',
    s_guiderule='G2',
    # s_guiderule='salor',
    s_test_dataset='voc12train_aug',
    s_test_datatype='Segmentation',
)


####

def parse_input(argv=sys.argv):
    parser = argparse.ArgumentParser(description="Trains a seed network")
    parser.add_argument('--init', default='VGG_ILSVRC_16_layers', type=str,
                        help='Initialisation for the network')
    parser.add_argument('--net', default='GAP-HighRes', type=str,
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
                             help='Cache intermediate and final results')
    parser_conf.add_argument('--overridecache', default=True, type=bool,
                             help='Override cache')
    conf = vars(parser_conf.parse_known_args(argv)[0])
    return control, conf


def write_proto(trainproto, testproto, conf, control):
    f = open(trainproto, 'w')
    f.write(deeplab(conf, control, 'train'))
    f.close()
    f = open(testproto, 'w')
    f.write(deeplab(conf, control, 'test'))
    f.close()
    return


def write_solver(solverproto, trainproto, testproto, learnedmodel_dir, control, conf):
    solverprototxt = tools.CaffeSolver(trainnet_prototxt_path=trainproto, testnet_prototxt_path=testproto)
    solverprototxt.sp['test_interval'] = "100000000"
    solverprototxt.sp['test_iter'] = "1"

    solverprototxt.sp['lr_policy'] = '"step"'
    solverprototxt.sp['gamma'] = "0.1"
    solverprototxt.sp['base_lr'] = str(control['base_lr'])
    solverprototxt.sp['display'] = "10"
    solverprototxt.sp['stepsize'] = conf['training']['stepsize']
    solverprototxt.sp['max_iter'] = conf['training']['max_iter']
    solverprototxt.sp['snapshot'] = conf['training']['snapshot']
    solverprototxt.sp['momentum'] = "0.9"
    solverprototxt.sp['weight_decay'] = "0.0005"
    solverprototxt.sp['snapshot_prefix'] = '"' + learnedmodel_dir + '/"'
    solverprototxt.sp['solver_mode'] = "GPU"
    solverprototxt.write(solverproto)

    return


def load_solver(control, control_token, conf):
    learnedmodel_dir = osp.join('cache', EXP_PHASE, create_token(control_token))
    mkdir_if_missing(learnedmodel_dir)
    print('saving to: {}'.format(learnedmodel_dir))

    # prototxt
    protodir = osp.join('models', EXP_PHASE, create_token(control_token))
    mkdir_if_missing(protodir)
    trainproto = osp.join(protodir, 'train.prototxt')
    testproto = osp.join(protodir, 'test.prototxt')
    solverproto = osp.join(protodir, 'solver.prototxt')
    write_proto(trainproto, testproto, conf, control)
    write_solver(solverproto, trainproto, testproto, learnedmodel_dir, control, conf)

    # init
    caffe.set_mode_gpu()
    caffe.set_device(conf['gpu'])
    solver = caffe.SGDSolver(solverproto)
    initmodel = osp.join('data', EXP_PHASE, control['init'] + '.caffemodel')
    solver.net.copy_from(initmodel)
    disp_net(solver.net)
    print('saving to: {}'.format(learnedmodel_dir))

    return solver


def main(control, conf):
    control, control_token, control_sup_token, conf = config_train(control, conf, EXP_PHASE)
    conf['sup_dir'] = osp.join('cache', 'guide-generate', create_token(control_sup_token))
    solver = load_solver(control, control_token, conf)
    solver.step(int(conf['training']['max_iter']))
    del solver


if __name__ == '__main__':
    if len(sys.argv) != 1:
        control, conf = parse_input(sys.argv)
    main(control, conf)
