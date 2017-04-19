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

from config import config_generate

####

EXP_PHASE = 'guide-generate'

conf = dict(
    vis=True,
    save=True,
    shuffle=True,
    overridecache=True,
    pascalroot="/BS/joon_projects/work",
    gpu=0,
    n=3,
    N=4,
)

control = dict(
    # seed
    g_init='VGG_ILSVRC_16_layers',
    g_net='GAP-HighRes',
    g_dataset='voc12train_aug',
    g_datatype='Segmentation',
    g_base_lr=0.001,
    g_batch_size=15,
    g_balbatch='clsbal',
    g_test_iter=8000,
    g_test_dataset='voc12train_aug',
    g_test_datatype='Segmentation',
    g_test_ranking='none',
    g_test_gtcls='use',

    # SAL
    s_net='DeepLabv2_ResNet',
    s_dataset='MSRA',
    s_datatype='NP',
    s_test_dataset='voc12train_aug',
    s_test_datatype='Segmentation',

    gtcls='use',
    seedthres=50,
    salthres=50,
    # guiderule='G0',
    # guiderule='G1',
    guiderule='G2',
    # guiderule='salor',
    test_dataset='voc12train_aug',
    test_datatype='Segmentation',
)


####


def parse_input(argv=sys.argv):
    parser = argparse.ArgumentParser(description="Generates guide ground truths for the segmentation network")
    parser.add_argument('--g_init', default='VGG_ILSVRC_16_layers', type=str,
                        help='Initialisation for the network')
    parser.add_argument('--g_net', default='GAP-HighRes', type=str,
                        help='Network')
    parser.add_argument('--g_dataset', default='voc12train_aug', type=str,
                        help='Training set')
    parser.add_argument('--g_datatype', default='Segmentation', type=str,
                        help='Type of training set')
    parser.add_argument('--g_base_lr', default=0.001, type=float,
                        help='Base learning rate')
    parser.add_argument('--g_batch_size', default=15, type=int,
                        help='Batch size')
    parser.add_argument('--g_balbatch', default='clsbal', type=str,
                        help='Class balanced batch composition')
    parser.add_argument('--g_test_iter', default=8000, type=int,
                        help='Test model iteration')
    parser.add_argument('--g_test_dataset', default='voc12train_aug', type=str,
                        help='Test dataset')
    parser.add_argument('--g_test_datatype', default='Segmentation', type=str,
                        help='Type of test data')
    parser.add_argument('--g_test_ranking', default='none', type=str,
                        help='When testing, dont rank priority according to size @ 20 percent max score as in lamperts')
    parser.add_argument('--g_test_gtcls', default='use', type=str,
                        help='Use GT class information at test time')

    parser.add_argument('--s_net', default='DeepLabv2_ResNet', type=str,
                        help='Network')
    parser.add_argument('--s_dataset', default='MSRA', type=str,
                        help='Training set')
    parser.add_argument('--s_datatype', default='NP', type=str,
                        help='Type of training set')
    parser.add_argument('--s_test_dataset', default='voc12train_aug', type=str,
                        help='Test dataset')
    parser.add_argument('--s_test_datatype', default='Segmentation', type=str,
                        help='Type of test data')

    parser.add_argument('--gtcls', default='use', type=str,
                        help='Use GT class information at test time')
    parser.add_argument('--seedthres', default=50, type=int,
                        help='FG threshold for seeds')
    parser.add_argument('--salthres', default=50, type=int,
                        help='FG threshold for saliency')
    parser.add_argument('--guiderule', default='G2', type=str,
                        help='Rule for generating guide labels')
    parser.add_argument('--test_dataset', default='voc12train_aug', type=str,
                        help='Test dataset')
    parser.add_argument('--test_datatype', default='Segmentation', type=str,
                        help='Type of test data')

    control = vars(parser.parse_known_args(argv)[0])

    parser_conf = argparse.ArgumentParser()
    parser_conf.add_argument('--pascalroot', default='/home', type=str,
                             help='Pascal VOC root folder')
    parser_conf.add_argument('--gpu', default=1, type=int,
                             help='GPU ID')
    parser_conf.add_argument('--vis', default=False, type=bool,
                             help='Visualisation')
    parser_conf.add_argument('--save', default=True, type=bool,
                             help='Save output')
    parser_conf.add_argument('--overridecache', default=True, type=bool,
                             help='Override cache')
    parser_conf.add_argument('--shuffle', default=True, type=bool,
                             help='Shuffle test input order')
    parser_conf.add_argument('--N', default=1, type=int,
                             help='Fragment test set into N fragments')
    parser_conf.add_argument('--n', default=0, type=int,
                             help='test n th fragment')
    conf = vars(parser_conf.parse_known_args(argv)[0])
    return control, conf


def save_guide(guide, im_id, conf):
    outfile = osp.join(conf['out_dir'], im_id + '.png')
    if not conf['overridecache']:
        assert (not os.path.isfile(outfile))
    else:
        if os.path.isfile(outfile):
            print('WARNING: OVERRIDING EXISTING RESULT FILE')
    cv2.imwrite(outfile, guide.astype(np.uint8))

    print('results saved to %s' % outfile)

    return


def mask_inference(image_original, gap_seed, ov_gap):
    involved_gtc = np.unique(np.array([o[0] for o in ov_gap]))
    unary = np.zeros((image_original.shape[0], image_original.shape[1], len(involved_gtc)), dtype=np.float)

    for ov_tmp in ov_gap:
        gtc, gc = ov_tmp
        heatmap = (gap_seed == gtc).astype(np.float32)
        heatmap = np.maximum(heatmap, 1e-3)
        heatmap = np.minimum(heatmap, 1 - 1e-3)
        heatmap = nd.filters.gaussian_filter(heatmap, 10)

        unary[:, :, np.where(gtc == involved_gtc)[0][0]] = heatmap

    unary_smoothed = CRF(image_original, unary, crf_param='deeplab')

    return involved_gtc[unary_smoothed.argmax(2)]


def combine_gap_sal(image_original, seed, saliency, gt_cls, control):
    imshape_original = image_original.shape[:2]

    if control['guiderule'] in ['G0']:
        raise NotImplementedError

    elif control['guiderule'] in ['G1']:
        raise NotImplementedError

    elif control['guiderule'] in ['G2', 'salor']:

        gap_cls_cc = {}
        gap_cls_cc_usage = {}
        for gtc in gt_cls:
            gap_cls_cc[gtc] = compute_cc(seed == gtc)
            gap_cls_cc_usage[gtc] = np.zeros(len(gap_cls_cc[gtc]))

        sal_cc = compute_cc(saliency, minarea=imshape_original[0] * imshape_original[1] * .01)

        seg = np.ones(imshape_original, dtype=np.uint8) * 255

        for sc in sal_cc:
            ov_gap = []
            for gtc in gt_cls:
                for idx, gc in enumerate(gap_cls_cc[gtc]):
                    ov = compute_iou(sc, gc)
                    if ov > 0:
                        if gtc == 0:
                            # if background, skip (overwrite gap bg with sal fg)
                            pass
                        else:
                            ov_gap.append([gtc, gc])
                        gap_cls_cc_usage[gtc][idx] = 1

            n_ovcls = len(np.unique(np.array([o[0] for o in ov_gap])))

            if n_ovcls == 0:  # no overlap with any GAP
                seg[sc] = 255
            elif n_ovcls == 1:
                seg[sc] = ov_gap[0][0]
            elif n_ovcls >= 2:
                pred = mask_inference(image_original, seed, ov_gap)
                seg[sc] = pred[sc]
            else:
                raise

        for gtc in gt_cls:
            for idx, gc in enumerate(gap_cls_cc[gtc]):
                if gap_cls_cc_usage[gtc][idx]:
                    if gtc == 0:
                        seg[np.maximum(0, gc - (gc & saliency)).astype(np.bool)] = 0

                else:
                    seg[gc] = gtc

        seg[seg == 255] = 0
    else:
        raise NotImplementedError

    return seg


def run_test_loop(control, conf):
    year = '20' + control['test_dataset'][3:5]
    pascal_list = get_pascal_indexlist(conf['pascalroot'], year, control['test_datatype'], control['test_dataset'][5:],
                                       shuffle=conf['shuffle'], n=conf['n'], N=conf['N'])
    num_test = len(pascal_list)
    print('%d images for testing' % num_test)

    start_time = time.time()
    for idx in range(num_test):
        end_time = time.time()
        print ('%d    Iter %d took %2.1f seconds' % (conf['n'], idx, end_time - start_time))
        start_time = time.time()
        print ('%d    Running %d out of %d images' % (conf['n'], idx + 1, num_test))

        inst = idx
        im_id = pascal_list[inst]
        outfile_save = osp.join(conf['out_dir'], im_id + '.png')

        if not conf['overridecache']:
            if conf['save']:
                if osp.isfile(outfile_save):
                    print('skipping')
                    continue

        imloc = conf['imgpath'] % im_id
        image = load_image_PIL(imloc)
        image_original = image.copy()

        gtfile = conf['clsimgpath'] % (im_id)
        gt = np.array(Image.open(gtfile)).astype(np.float)
        gt_cls = np.unique(gt)
        gt_cls_nobg = gt_cls[(gt_cls != 0) & (gt_cls != 255)]

        # Load seed
        ld = sio.loadmat(osp.join(conf['seed_dir'], im_id + '.mat'))
        seed_heatmap = ld['heatmap'].astype(np.float)
        imshape_original = ld['imshape_original'][0]
        seed_argmax, confidence = heatmap2segconf(seed_heatmap, imshape_original, gt_cls_nobg)
        seed = seed_argmax * (confidence * 100 >= control['seedthres'])

        # load saliency
        if 'salor' in control['guiderule']:
            sal_map = ((gt != 0) & (gt != 255)).astype(np.float) * 255
        else:
            sal_map = load_image_PIL(osp.join(conf['saliency_dir'], im_id + '.png')).astype(np.float)
        saliency = sal_map >= control['salthres'] * 255. / 100

        guide = combine_gap_sal(image_original, seed, saliency, gt_cls, control)

        def visualise_data():
            fig = plt.figure(3, figsize=(15, 10))
            fig.suptitle('ID:{}'.format(im_id))
            ax = fig.add_subplot(2, 5, 1)
            ax.set_title('Original image')
            pim(image_original)
            ax = fig.add_subplot(2, 5, 6)
            ax.imshow(image_original)
            ax.imshow(gt, alpha=.5, cmap="nipy_spectral", clim=(0, 30))

            ax = fig.add_subplot(2, 5, 2)
            ax.set_title('Seed')
            ax.imshow(seed, cmap="nipy_spectral", clim=(0, 30))
            ax = fig.add_subplot(2, 5, 7)
            ax.imshow(image_original)
            ax.imshow(seed, alpha=.5, cmap="nipy_spectral", clim=(0, 30))

            ax = fig.add_subplot(2, 5, 3)
            ax.set_title('Saliency')
            ax.imshow(saliency, cmap="hot", clim=(0, 1.5))
            ax = fig.add_subplot(2, 5, 8)
            ax.imshow(image_original)
            ax.imshow(saliency, alpha=.5, cmap="hot", clim=(0, 1.5))

            ax = fig.add_subplot(2, 5, 4)
            ax.set_title('Together')
            ax.imshow(saliency, cmap="hot", clim=(0, 1.5))
            ax.imshow(seed, alpha=.5, cmap="nipy_spectral", clim=(0, 30))
            ax = fig.add_subplot(2, 5, 9)
            ax.imshow(image_original)
            ax.imshow(saliency, alpha=.3, cmap="hot", clim=(0, 1.5))
            ax.imshow(seed, alpha=.5, cmap="nipy_spectral", clim=(0, 30))

            ax = fig.add_subplot(2, 5, 5)
            ax.set_title('Guide (%s)' % control['guiderule'])
            ax.imshow(guide, cmap="nipy_spectral", clim=(0, 30))
            ax = fig.add_subplot(2, 5, 10)
            ax.imshow(image_original)
            ax.imshow(guide, alpha=.5, cmap="nipy_spectral", clim=(0, 30))

            for iii in range(10):
                fig.axes[iii].get_xaxis().set_visible(False)
                fig.axes[iii].get_yaxis().set_visible(False)

            plt.pause(1)

            return

        if conf['vis']:
            visualise_data()

        if conf['save']:
            save_guide(guide, im_id, conf)

    return


def main(control, conf):
    control, control_token, control_seed, control_seed_token, control_saliency, control_saliency_token, conf = \
        config_generate(control, conf, EXP_PHASE)

    conf['out_dir'] = osp.join('cache', EXP_PHASE, create_token(control_token))
    mkdir_if_missing(conf['out_dir'])
    print('saving to: {}'.format(conf['out_dir']))

    conf['seed_dir'] = osp.join('cache', 'seed-test', create_token(control_seed_token))
    conf['saliency_dir'] = osp.join('cache', 'saliency-test', create_token(control_saliency_token))

    run_test_loop(control, conf)


if __name__ == '__main__':
    if len(sys.argv) != 1:
        control, conf = parse_input(sys.argv)
    main(control, conf)
