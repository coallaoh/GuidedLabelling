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

####

EXP_PHASE = 'saliency-test'

conf = dict(
    vis=True,
    save_heat=True,
    overridecache=True,
    shuffle=False,
    pascalroot="/BS/joon_projects/work",
    gpu=1,
    n=0,
    N=1,
)

control = dict(
    net='DeepLabv2_ResNet',
    dataset='MSRA',
    datatype='NP',
    test_dataset='voc12val',
    test_datatype='Segmentation',
)


####

def parse_input(argv=sys.argv):
    parser = argparse.ArgumentParser(description="Trains a seed network")
    parser.add_argument('--net', default='DeepLabv2_ResNet', type=str,
                        help='Network')
    parser.add_argument('--dataset', default='MSRA', type=str,
                        help='Training set')
    parser.add_argument('--datatype', default='NP', type=str,
                        help='Type of training set')
    parser.add_argument('--test_dataset', default='voc12val', type=str,
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
    parser_conf.add_argument('--save_heat', default=True, type=bool,
                             help='Save raw heatmap output')
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


def process_test_input(im, transformer, input_padded_size):
    H, W = im.shape[:2]
    if (H <= input_padded_size and W <= input_padded_size):
        factor = 1
        pass
    else:
        factor = min(float(input_padded_size) / H, float(input_padded_size) / W)
        im = scipy.misc.imresize(im, factor, interp="bilinear")
        H, W = im.shape[:2]
    margin = [input_padded_size - H, input_padded_size - W]
    margins = [
        margin[0] // 2,
        margin[0] - margin[0] // 2,
        margin[1] // 2,
        margin[1] - margin[1] // 2,
    ]
    input_image = cv2.copyMakeBorder(im, margins[0], margins[1], margins[2], margins[3], cv2.BORDER_REFLECT_101)
    input_image = transformer.preprocess('data', input_image)

    return input_image, margins, factor, H, W


def net_forward(net, input_image):
    net.blobs['data'].data[:] = input_image.astype(np.float32)
    output = net.forward()
    prob = output['fc1_prob'][0]
    return prob


def process_test_output_prob(prob, margins, H_original, W_original, H, W, input_padded_size, factor):
    prob_trimmed = np.zeros([2, H_original, W_original], dtype=np.float32)

    for ch in range(prob_trimmed.shape[0]):
        prob_tmp = scipy.misc.imresize(prob[ch], [input_padded_size, input_padded_size], mode='F')
        prob_tmp = prob_tmp[margins[0]:margins[0] + H, margins[2]:margins[2] + W]
        prob_trimmed[ch] = scipy.misc.imresize(prob_tmp, [H_original, W_original], mode='F')

    return prob_trimmed


def run_test(net, out_dir, control, conf):
    year = '20' + control['test_dataset'][3:5]
    pascal_list = get_pascal_indexlist(conf['pascalroot'], year, control['test_datatype'], control['test_dataset'][5:],
                                       shuffle=conf['shuffle'])

    num_test = len(pascal_list)
    start_idx = int(np.round(conf['n'] * num_test / float(conf['N'])))
    end_idx = int(np.round((conf['n'] + 1) * num_test / float(conf['N'])))
    pascal_list = pascal_list[start_idx:end_idx]
    num_test = len(pascal_list)

    print('%d images for testing' % num_test)

    transformer = set_preprocessor_without_net(
        [1, 3, conf['input_padded_size'], conf['input_padded_size']],
        mean_image=np.array([104.008, 116.669, 122.675],
                            dtype=np.float))

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
        outfile = osp.join(out_dir, im_id + '.mat')

        if conf['save_heat']:
            if not conf['overridecache']:
                if osp.isfile(outfile):
                    print('skipping')
                    continue

        imloc = os.path.join(conf['pascalroot'], 'VOC' + year, 'JPEGImages', im_id + '.jpg')
        image = load_image_PIL(imloc)
        imshape_original = image.shape[:2]

        input_image, margins, factor, H, W = process_test_input(image, transformer, conf['input_padded_size'])
        prob = net_forward(net, input_image)

        prob = process_test_output_prob(prob, margins, imshape_original[0], imshape_original[1], H, W,
                                        conf['input_padded_size'], factor)

        sal = 255 * prob[1] / prob[1].max()

        if conf['vis']:
            def visualise_data():
                fig = plt.figure(0, figsize=(15, 10))
                fig.suptitle('ID:{}'.format(im_id))
                ax = fig.add_subplot(1, 2, 1)
                ax.set_title('Original image')
                pim(image)

                ax = fig.add_subplot(2, 2, 2)
                ax.set_title('Saliency prediction')
                ax.imshow(sal, cmap="hot", clim=(0, 255))
                ax = fig.add_subplot(2, 2, 4)
                ax.imshow(image)
                ax.imshow(sal, alpha=.5, cmap="hot", clim=(0, 255))

                for iii in range(3):
                    fig.axes[iii].get_xaxis().set_visible(False)
                    fig.axes[iii].get_yaxis().set_visible(False)

                plt.pause(1)
                return

            visualise_data()

        if conf['save_heat']:
            if not conf['overridecache']:
                assert (not os.path.isfile(outfile))
            else:
                if os.path.isfile(outfile):
                    print('WARNING: OVERRIDING EXISTING RESULT FILE')
            sio.savemat(outfile, dict(heatmap=sal, imshape_original=imshape_original))
            print('results saved to %s' % outfile)

    return


def crawl_net(conf):
    testproto = osp.join('data', EXP_PHASE, 'deploy.prototxt')
    learnedmodel = osp.join('data', EXP_PHASE, 'weights.caffemodel')

    caffe.set_mode_gpu()
    caffe.set_device(conf['gpu'])
    net = caffe.Net(testproto, learnedmodel, caffe.TEST)

    return net


def main(control, conf):
    control, control_token, conf = config_test(control, conf, EXP_PHASE)

    out_dir = osp.join('cache', EXP_PHASE, create_token(control_token))
    mkdir_if_missing(out_dir)
    print('saving to: {}'.format(out_dir))

    net = crawl_net(conf)

    run_test(net, out_dir, control, conf)


if __name__ == '__main__':
    if len(sys.argv) != 1:
        control, conf = parse_input(sys.argv)
    main(control, conf)
