#!/usr/bin/env python

__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'ResearchTools')

from imports.basic_modules import *
from imports.ResearchTools import *

# Pipeline of the algorithm
TRAIN_SEED = True
TEST_SEED = True
LOAD_SALIENCY = True
GENERATE_GUIDE = True
TRAIN_SEG = True
TEST_SEG = True

# Common options
PASCALROOT = "/BS/joon_projects/work/"
VISUALISE = False
SAVE_CACHE = True
GPU = 1

SEED_TYPE = 'GAP-HighRes'
SEED_THRES = 20

if TRAIN_SEED:
    print("##########\nSeed Training\n##########")
    from seed.train import main as seed_train

    conf = dict(
        vis=VISUALISE,
        save=SAVE_CACHE,
        overridecache=True,
        pascalroot=PASCALROOT,
        imagenetmeanloc="data/ilsvrc_2012_mean.npy",
        gpu=GPU,
    )

    control = dict(
        init='VGG_ILSVRC_16_layers',
        net=SEED_TYPE,
        dataset='voc12train_aug',
        datatype='Segmentation',
        base_lr=0.001,
        batch_size=15,
        balbatch='clsbal',
    )

    seed_train(control, conf)

if TEST_SEED:
    print("##########\nSeed Testing\n##########")
    from seed.test import main as seed_test

    conf = dict(
        save_cls=SAVE_CACHE,
        save_heat=SAVE_CACHE,
        vis=VISUALISE,
        visconf=0.5,
        shuffle=True,
        overridecache=True,
        pascalroot=PASCALROOT,
        imagenetmeanloc="data/ilsvrc_2012_mean.npy",
        gpu=GPU,
    )

    control = dict(
        init='VGG_ILSVRC_16_layers',
        net=SEED_TYPE,
        dataset='voc12train_aug',
        datatype='Segmentation',
        base_lr=0.001,
        batch_size=15,
        balbatch='clsbal',
        test_iter=8000,
        test_dataset='voc12train_aug',
        test_datatype='Segmentation',
        test_ranking='none',
        test_interpord=1,
        test_gtcls='use',
    )
    seed_test(control, conf)

if LOAD_SALIENCY:
    print("##########\nCopying Saliency Output\n##########")

    mkdir_if_missing("cache/saliency-test/_test_dataset:voc12train_aug")
    os.system(
        "tar xf data/saliency-test/test_dataset:voc12train_aug.tar.gz -C cache/saliency-test/_test_dataset:voc12train_aug/ --force-local")

if GENERATE_GUIDE:
    print("##########\nGenerating Guide\n##########")
    from guide_generation.generate import main as generate_guide

    conf = dict(
        vis=VISUALISE,
        save=SAVE_CACHE,
        shuffle=True,
        overridecache=True,
        pascalroot=PASCALROOT,
        gpu=GPU,
        n=0,
        N=1,
    )

    control = dict(
        # seed
        g_init='VGG_ILSVRC_16_layers',
        g_net=SEED_TYPE,
        g_dataset='voc12train_aug',
        g_datatype='Segmentation',
        g_base_lr=0.001,
        g_batch_size=15,
        g_balbatch='clsbal',
        g_test_iter=8000,
        g_test_dataset='voc12train_aug',
        g_test_datatype='Segmentation',
        g_test_ranking='none',
        g_test_interpord=1,
        g_test_gtcls='use',

        # SAL
        s_net='DeepLabv2_ResNet',
        s_dataset='MSRA',
        s_datatype='NP',
        s_test_dataset='voc12train_aug',
        s_test_datatype='Segmentation',

        gtcls='use',
        seedthres=SEED_THRES,
        salthres=50,
        guiderule='G2',
        test_dataset='voc12train_aug',
        test_datatype='Segmentation',
    )

    generate_guide(control, conf)

if TRAIN_SEG:
    print("##########\nTraining Segmenter\n##########")
    from segmentation.train import main as seg_train

    conf = dict(
        vis=VISUALISE,
        save=SAVE_CACHE,
        overridecache=True,
        pascalroot=PASCALROOT,
        imagenetmeanloc="data/ilsvrc_2012_mean.npy",
        gpu=GPU,
    )

    control = dict(
        init='VGG_ILSVRC_16_layers-deeplab',
        net='DeepLab',
        dataset='voc12train_aug',
        datatype='Segmentation',
        base_lr=0.001,
        batch_size=15,
        resize='none',

        # seed
        s_g_init='VGG_ILSVRC_16_layers',
        s_g_net=SEED_TYPE,
        s_g_dataset='voc12train_aug',
        s_g_datatype='Segmentation',
        s_g_base_lr=0.001,
        s_g_batch_size=15,
        s_g_balbatch='clsbal',
        s_g_test_iter=8000,
        s_g_test_dataset='voc12train_aug',
        s_g_test_datatype='Segmentation',
        s_g_test_ranking='none',
        s_g_test_interpord=1,
        s_g_test_gtcls='use',

        # SAL
        s_s_net='DeepLabv2_ResNet',
        s_s_dataset='MSRA',
        s_s_datatype='NP',
        s_s_test_dataset='voc12train_aug',
        s_s_test_datatype='Segmentation',

        s_gtcls='use',
        s_seedthres=SEED_THRES,
        s_salthres=50,
        s_guiderule='G2',
        s_test_dataset='voc12train_aug',
        s_test_datatype='Segmentation',
    )

    seg_train(control, conf)

if TEST_SEG:
    print("##########\nTesting Segmenter\n##########")
    from segmentation.test import main as seg_test

    conf = dict(
        save=SAVE_CACHE,
        vis=VISUALISE,
        shuffle=True,
        overridecache=True,
        pascalroot=PASCALROOT,
        imagenetmeanloc="data/ilsvrc_2012_mean.npy",
        gpu=GPU,
    )

    control = dict(
        init='VGG_ILSVRC_16_layers-deeplab',
        net='DeepLab',
        dataset='voc12train_aug',
        datatype='Segmentation',
        base_lr=0.001,
        batch_size=15,
        resize='none',

        # seed
        s_g_init='VGG_ILSVRC_16_layers',
        s_g_net=SEED_TYPE,
        s_g_dataset='voc12train_aug',
        s_g_datatype='Segmentation',
        s_g_base_lr=0.001,
        s_g_batch_size=15,
        s_g_balbatch='clsbal',
        s_g_test_iter=8000,
        s_g_test_dataset='voc12train_aug',
        s_g_test_datatype='Segmentation',
        s_g_test_ranking='none',
        s_g_test_interpord=1,
        s_g_test_gtcls='use',

        # SAL
        s_s_net='DeepLabv2_ResNet',
        s_s_dataset='MSRA',
        s_s_datatype='NP',
        s_s_test_dataset='voc12train_aug',
        s_s_test_datatype='Segmentation',

        s_gtcls='use',
        s_seedthres=SEED_THRES,
        s_salthres=50,
        s_guiderule='G2',
        s_test_dataset='voc12train_aug',
        s_test_datatype='Segmentation',

        test_iter=8000,
        test_dataset='voc12val',
        test_datatype='Segmentation',
        test_pcrf='deeplab',
        test_resize='none',
    )

    seg_test(control, conf)
