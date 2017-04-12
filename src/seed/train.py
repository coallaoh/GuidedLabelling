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

from config import config_train
from networks import gap

####

EXP_PHASE = 'seed-train'

conf = dict(
    vis=False,
    save=True,
    overridecache=True,
    pascalroot="/BS/joon_projects/work/",
    imagenetmeanloc="data/ilsvrc_2012_mean.npy",
    gpu=2,
)

control = dict(
    init='VGG_ILSVRC_16_layers',
    net='GAP-DeepLab',
    dataset='voc12train_aug',
    datatype='Segmentation',
    base_lr=0.001,
    batch_size=15,
    balbatch='clsbal',
)


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
    parser.add_argument('--balbatch', default='clsbal', type=str,
                        help='Class balanced batch composition')
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
    f.write(gap(conf, control, 'train'))
    f.close()
    f = open(testproto, 'w')
    f.write(gap(conf, control, 'test'))
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


def get_initmodel(control):
    if control['net'] == 'GAP-LowRes':
        initmodel = osp.join('data', EXP_PHASE, control['init'] + '.caffemodel')
    elif control['net'] == 'GAP-HighRes':
        initmodel = osp.join('data', EXP_PHASE, control['init'] + '.caffemodel')
    elif control['net'] == 'GAP-ROI':
        initmodel = osp.join('data', EXP_PHASE, control['init'] + '.caffemodel')
    elif control['net'] == 'GAP-DeepLab':
        initmodel = osp.join('data', 'seed-train', control['init'] + '-deeplab.caffemodel')
    else:
        raise NotImplementedError
    return initmodel


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
    initmodel = get_initmodel(control)
    solver.net.copy_from(initmodel)
    disp_net(solver.net)
    print('saving to: {}'.format(learnedmodel_dir))

    return solver


def main(control, conf):
    control, control_token, conf = config_train(control, conf, EXP_PHASE)
    solver = load_solver(control, control_token, conf)
    solver.step(int(conf['training']['max_iter']))
    del solver


if __name__ == "__main__":
    if len(sys.argv) != 1:
        control, conf = parse_input(sys.argv)
    main(control, conf)
