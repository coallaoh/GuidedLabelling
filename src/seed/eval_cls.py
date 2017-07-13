#!/usr/bin/env python

__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
sys.path.insert(0, 'ResearchTools')

from imports.basic_modules import *
from imports.ResearchTools import *
from imports.libmodules import *

from config import config_eval_cls

###

EXP_PHASE = 'seed-eval-cls'

conf = dict(
    pascalroot="/BS/joon_projects/work/",
)

control = dict(
    init='VGG_ILSVRC_16_layers',
    net='GAP-HighRes',
    dataset='voc12train_aug',
    datatype='Segmentation',
    base_lr=0.001,
    batch_size=15,
    balbatch='clsbal',
    test_iter=8000,
    test_dataset='voc12val',
    test_datatype='Main',
    test_ranking='none',
    test_interpord=1,
    test_gtcls='use',
)


####

def parse_input(argv=sys.argv):
    parser = argparse.ArgumentParser(description="Evaluate a seed network as classifier (mAP)")
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
    parser.add_argument('--test_iter', default=8000, type=int,
                        help='Test model iteration')
    parser.add_argument('--test_dataset', default='voc12val', type=str,
                        help='Test dataset')
    parser.add_argument('--test_datatype', default='Main', type=str,
                        help='Type of test data')
    parser.add_argument('--test_ranking', default='none', type=str,
                        help='When testing, dont rank priority according to size @ 20 percent max score as in lamperts')
    parser.add_argument('--test_interpord', default=1, type=int,
                        help='Interpolation order')
    parser.add_argument('--test_gtcls', default='use', type=str,
                        help='Use GT class information at test time')
    control = vars(parser.parse_known_args(argv)[0])

    parser_conf = argparse.ArgumentParser()
    parser_conf.add_argument('--pascalroot', default='/home', type=str,
                             help='Pascal VOC root folder')
    conf = vars(parser_conf.parse_known_args(argv)[0])
    return control, conf


def cls_mAP(conf):
    def VOCap(rec, prec):
        mrec = rec.copy()
        mrec = np.insert(mrec, 0, [0])
        mrec = np.append(mrec, [1])
        mpre = prec.copy()
        mpre = np.insert(mpre, 0, [0])
        mpre = np.append(mpre, [0])
        # for i=numel(mpre)-2:-1:0   :
        for i in xrange(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1]).copy()
        i = np.where(mrec[1:] != mrec[:-1])[0] + 1
        ap = ((mrec[i] - mrec[i - 1]) * mpre[i]).sum()
        return ap

    ld = sio.loadmat(conf['cls_file'])
    res_ids = ld['ids']
    confidence = ld['res']  # Nx20 matrix

    ap_list = []
    for cls_idx, clsname in enumerate(get_pascal_classes()):
        gtfile = open(conf['clsgt'] % clsname)
        gt_list = gtfile.readlines()
        gtids = [gt_item.split()[0] for gt_item in gt_list]
        gt = np.array([int(gt_item.split()[1].rstrip('\n')) for gt_item in gt_list])

        assert ((np.array(gtids) == res_ids).all())

        si = np.argsort(-confidence[:, cls_idx])
        tp = gt[si] > 0
        fp = gt[si] < 0

        fp = np.cumsum(fp).astype(np.float)
        tp = np.cumsum(tp).astype(np.float)
        rec = tp / (gt > 0).sum()
        prec = tp / (fp + tp)

        ap = VOCap(rec, prec)
        ap_list.append(ap)
        print('     AP of %8s: %2.2f' % (clsname, ap * 100))

    print('mAP: %1.6f' % np.array(ap_list).mean())
    return


def main(control, conf):
    control, control_token, conf = config_eval_cls(control, conf, EXP_PHASE)
    result_dir = osp.join('cache', 'seed-test', create_token(control_token))
    conf['cls_file'] = osp.join(result_dir, 'cls_res.mat')
    cls_mAP(conf)
    return


if __name__ == "__main__":
    if len(sys.argv) != 1:
        control, conf = parse_input(sys.argv)
    main(control, conf)
