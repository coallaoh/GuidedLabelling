#!/usr/bin/env python

__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
sys.path.insert(0, 'ResearchTools')

from imports.basic_modules import *
from imports.ResearchTools import *
from imports.libmodules import *

from config import config_eval

###

EXP_PHASE = 'seed-eval'

conf = dict(
    overridecache=False,
    save=True,
    parallel=40,
    pascalroot="/BS/joon_projects/work/",
)

control = dict(
    init='VGG_ILSVRC_16_layers',
    net='GAP-DeepLab',
    dataset='voc12train_aug',
    datatype='Segmentation',
    base_lr=0.001,
    batch_size=15,
    balbatch='clsbal',
    test_iter=8000,
    test_dataset='voc12val',
    test_datatype='Segmentation',
    test_ranking='none',
    test_gtcls='use',
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
    parser.add_argument('--balbatch', default='clsbal', type=str,
                        help='Class balanced batch composition')
    parser.add_argument('--test_iter', default=8000, type=int,
                        help='Test model iteration')
    parser.add_argument('--test_dataset', default='voc12val', type=str,
                        help='Test dataset')
    parser.add_argument('--test_datatype', default='Segmentation', type=str,
                        help='Type of test data')
    parser.add_argument('--test_ranking', default='none', type=str,
                        help='When testing, dont rank priority according to size @ 20% max score as in lamperts')
    parser.add_argument('--test_gtcls', default='use', type=str,
                        help='Use GT class information at test time')
    control = vars(parser.parse_known_args(argv)[0])

    parser_conf = argparse.ArgumentParser()
    parser_conf.add_argument('--pascalroot', default='/home', type=str,
                             help='Pascal VOC root folder')
    parser_conf.add_argument('--save', default=True, type=bool,
                             help='Save evaluation results')
    parser_conf.add_argument('--overridecache', default=True, type=bool,
                             help='Override cache')
    parser_conf.add_argument('--parallel', default=1, type=int,
                             help='Number of cores for parallel computation')
    conf = vars(parser_conf.parse_known_args(argv)[0])
    return control, conf


def compute_pred_hist(*args):
    gtids, idx, conf = args[0]
    num = conf['nclasses'] + 1
    thresrange = conf['thresrange']
    print('test confusion: %d/%d' % (idx + 1, len(gtids)))
    imname = gtids[idx]

    # ground truth label file
    gtfile = conf['clsimgpath'] % (imname)
    gtim = np.array(Image.open(gtfile)).astype(np.float)

    # heatmap file
    heatfile = conf['result_file'] % imname
    ld = sio.loadmat(heatfile)
    heatmap = ld['heatmap'].astype(np.float)
    imshape_original = ld['imshape_original'][0]

    # results file
    gt_cls = np.unique(gtim)
    gt_cls = gt_cls[gt_cls != 0]
    gt_cls = gt_cls[gt_cls != 255]
    resim, confim = heatmap2segconf(heatmap, imshape_original, gt_cls)
    resim[resim == 255] = 21

    if np.isnan(confim).any():
        print('Warning: confidence map has value nan')
        confim[np.isnan(confim)] = thresrange.min()

    if conf['threstype'] == 'perim':
        confim = confim - confim.min()
        confim = confim / (confim.max() - confim.min())
        assert (confim.min() == 0 and confim.max() == 1)

    elif conf['threstype'] == 'generic':
        confim[confim >= thresrange.max()] = thresrange.max()
        confim[confim <= thresrange.min()] = thresrange.min()
        assert (confim.min() >= thresrange.min() and confim.max() <= thresrange.max())

    # Check validity of results image
    maxlabel = resim.max()
    if maxlabel > 255:
        raise Exception('Results image ''%s'' has out of range value %d (the value should be <= %d)' % (
            imname, maxlabel, conf['nclasses']))

    szgtim = gtim.shape
    szresim = resim.shape
    if szgtim != szresim:
        raise Exception('Results image ''%s'' is the wrong size, was %d x %d, should be %d x %d.' % (
            imname, szresim[0], szresim[1], szgtim[0], szgtim[1]))

    # pixel locations to include in computation
    locs = (gtim < 255)

    # joint histogram
    sumim = gtim + resim * num
    # hs = np.histogram(sumim[locs], range(num ** 2 + 1))[0].reshape((num, -1)).T

    confidencehists_local = np.zeros((len(thresrange), num, num + 1), dtype=np.int)

    # confidence histogram
    for gtj in range(num):
        for resj in range(num + 1):
            entry = gtj + resj * num

            entrylocs = sumim == entry
            entryconfs = confim[entrylocs & locs]

            thresrange_ = np.append(thresrange, thresrange[-1] + thresrange[-1] - thresrange[-2])
            confidencecnts = np.histogram(entryconfs, thresrange_)[0].astype(np.int)
            confidencecnts_padded = np.zeros((len(thresrange), num, num + 1), dtype=np.int)
            confidencecnts_padded[:, gtj, resj] = confidencecnts
            confidencehists_local = confidencehists_local + confidencecnts_padded

    return confidencehists_local


def compute_basic_counts(confidencehists, conf):
    num = conf['nclasses'] + 1
    confcounts = confidencehists.sum(axis=0)
    accuracies = np.zeros(num)
    iog = np.zeros(num)
    iop = np.zeros(num)
    segsize = np.zeros(num)
    print('Accuracy for each class (intersection/union measure)')
    for j in range(num):
        gtj = confcounts[j, :].sum().astype(np.float)
        resj = confcounts[:, j].sum().astype(np.float)
        gtjresj = confcounts[j, j].astype(np.float)

        accuracies[j] = 100 * gtjresj / (gtj + resj - gtjresj)
        iog[j] = 100 * gtjresj / (gtj)
        iop[j] = 100 * gtjresj / (resj)
        segsize[j] = (resj - gtjresj) / (gtj - gtjresj)

        print('  %14s: %6.1f%%' % (str(j), accuracies[j]))

    avacc = accuracies.mean()
    print('-------------------------')
    return dict(avacc=avacc, confcounts=confcounts, accuracies=accuracies, iog=iog, iop=iop, segsize=segsize)


def get_curves(confidencehists, results, conf):
    confcounts = results['confcounts']
    thresrange = conf['thresrange']
    num = conf['nclasses'] + 1
    confidencehists_cum = np.cumsum(confidencehists, axis=0)

    acc = np.zeros((len(thresrange) + 1, num))
    prec = np.zeros((len(thresrange) + 1, num))
    rec = np.zeros((len(thresrange) + 1, num))

    for i in range(len(thresrange) + 1):
        confcounts_thres = confcounts.copy()
        for j_gt in range(1, num + 1):
            for j_pred in range(2, num + 1):
                BG = 0
                if i > 0:
                    BG = confidencehists_cum[i - 1, j_gt - 1, j_pred - 1]
                confcounts_thres[j_gt - 1, j_pred - 1] = confcounts_thres[j_gt - 1, j_pred - 1] - BG
                confcounts_thres[j_gt - 1, 0] = confcounts_thres[j_gt - 1, 0] + BG

        for j in range(1, num + 1):  # 1:num
            gtj = confcounts_thres[j - 1, :].sum().astype(np.float)
            resj = confcounts_thres[:, j - 1].sum().astype(np.float)
            gtjresj = confcounts_thres[j - 1, j - 1].astype(np.float)

            prec[i, j - 1] = gtjresj / resj
            rec[i, j - 1] = gtjresj / gtj
            acc[i, j - 1] = gtjresj / (gtj + resj - gtjresj)

    results['prec'] = prec
    results['rec'] = rec
    results['acc'] = acc
    return


def compute_confidencehists(conf):
    gtids = get_pascal_indexlist(conf['pascalroot'], conf['year'], control['test_datatype'], conf['testset'])
    num = conf['nclasses'] + 1
    confidencehists_ = Sum((len(conf['thresrange']), num, num + 1))
    if conf['parallel'] > 1:
        pool = multiprocessing.Pool(processes=conf['parallel'])
        for idx in range(len(gtids)):
            pool.apply_async(apply_async_wrapper, (
                compute_pred_hist, (gtids, idx, conf)
            ), callback=confidencehists_.add)
        pool.close()
        pool.join()
    else:
        for idx in range(len(gtids)):
            confidencehists_.add(compute_pred_hist((gtids, idx, conf)))

    return confidencehists_.value


def report_fg_bg_prec_rec(prec, rec):
    print("FG PREC: %4.4f" % prec[2, 1:].mean())
    print("FG REC : %4.4f" % rec[2, 1:].mean())
    print("BG PREC: %4.4f" % prec[2, 0])
    print("BG REC : %4.4f" % rec[2, 0])

    return


def print_results(statsfile):
    ld = sio.loadmat(statsfile)
    prec = ld['results'][0][0]['prec']
    rec = ld['results'][0][0]['rec']
    acc = ld['results'][0][0]['acc']
    pprint.pprint(ld)
    report_fg_bg_prec_rec(prec, rec)

    print("Maximal mIoU: %2.2f" % (acc.mean(1).max() * 100))
    print("Prec:")
    print(prec.mean(1))
    print("Rec:")
    print(rec.mean(1))
    return


def compute_precrec(conf):
    confidencehists = compute_confidencehists(conf)
    results = compute_basic_counts(confidencehists, conf)
    get_curves(confidencehists, results, conf)

    data = dict(
        control=control,
        conf=conf,
        results=results,
    )
    if conf['save']:
        sio.savemat(conf['stats_file'], data)
    return


def main(control, conf):
    control, control_token, conf = config_eval(control, conf, EXP_PHASE)
    result_dir = osp.join('cache', 'seed-test', create_token(control_token))
    conf['result_file'] = osp.join(result_dir, '%s.mat')
    conf['stats_file'] = osp.join(result_dir, 'mIoU.py.mat')

    if not conf['overridecache']:
        if osp.isfile(conf['stats_file']):
            print_results(conf['stats_file'])
            return

    compute_precrec(conf)
    print_results(conf['stats_file'])
    return


if __name__ == "__main__":
    if len(sys.argv) != 1:
        control, conf = parse_input(sys.argv)
    main(control, conf)
