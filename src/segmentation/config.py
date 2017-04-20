__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
sys.path.insert(0, 'ResearchTools')

from imports.basic_modules import *
from imports.ResearchTools import *
from imports.libmodules import *


def config_train(control, conf, EXP_PHASE):
    assert (EXP_PHASE == 'seg-train')

    assert (control['s_g_test_dataset'] == control['s_s_test_dataset'])
    assert (control['s_test_dataset'] == control['s_s_test_dataset'])
    assert (control['s_g_test_datatype'] == control['s_s_test_datatype'])
    assert (control['s_test_datatype'] == control['s_s_test_datatype'])
    assert (control['dataset'] == control['s_s_test_dataset'])
    assert (control['datatype'] == control['s_s_test_datatype'])

    control_sup = subcontrol(control, 's')

    from guide_generation.config import config_generate

    control_sup, control_sup_token, _, _, _, _, _ = config_generate(control_sup, conf, 'guide-generate')

    default_control = dict(
        init='VGG_ILSVRC_16_layers-deeplab',
        net='DeepLab',
        dataset='voc12train_aug',
        datatype='Segmentation',
        base_lr=0.001,
        batch_size=15,

        # seed
        s_g_init='VGG_ILSVRC_16_layers',
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
        s_test_dataset='voc12train_aug',
        s_test_datatype='Segmentation',
    )

    control_token = control.copy()
    for ky in default_control:
        if ky in control_token.keys():
            if control_token[ky] == default_control[ky]:
                control_token.pop(ky)

    conf['EXP_PHASE'] = EXP_PHASE

    # 11-12 epochs
    conf['training'] = dict(
        stepsize="2000",
        max_iter="8000",
        snapshot="8000",
    )

    conf['input_size'] = 321
    conf['output_size'] = 41
    if 'voc' in control['dataset']:
        conf['nposcls'] = 20
    else:
        raise NotImplementedError

    pprint.pprint(conf)
    pprint.pprint(control)

    return control, control_token, control_sup_token, conf


def config_test(control, conf, EXP_PHASE):
    assert (EXP_PHASE == 'seg-test')

    defaults = dict(
        init='VGG_ILSVRC_16_layers-deeplab',
        net='DeepLab',
        dataset='voc12train_aug',
        datatype='Segmentation',
        base_lr=0.001,
        batch_size=15,

        # seed
        s_g_init='VGG_ILSVRC_16_layers',
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
        s_test_dataset='voc12train_aug',
        s_test_datatype='Segmentation',

        test_iter=8000,
        test_datatype='Segmentation',
        test_pcrf='none',
    )

    defaults_model = dict(
        init='VGG_ILSVRC_16_layers-deeplab',
        net='DeepLab',
        dataset='voc12train_aug',
        datatype='Segmentation',
        base_lr=0.001,
        batch_size=15,

        # seed
        s_g_init='VGG_ILSVRC_16_layers',
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
        s_test_dataset='voc12train_aug',
        s_test_datatype='Segmentation',
    )

    exclude_from_model = [
        'test_iter',
        'test_dataset',
        'test_datatype',
        'test_pcrf',
    ]

    control_model = control.copy()
    for ky in defaults_model:
        if control_model[ky] == defaults_model[ky]:
            control_model.pop(ky)

    for exc in exclude_from_model:
        if exc in control_model.keys():
            control_model.pop(exc)

    control_token = control.copy()
    for ky in defaults:
        if control_token[ky] == defaults[ky]:
            control_token.pop(ky)

    conf['EXP_PHASE'] = EXP_PHASE

    conf['input_size'] = 321
    conf['output_size'] = 41
    if 'voc' in control['dataset']:
        conf['nposcls'] = 20
    else:
        raise NotImplementedError

    load_pascal_conf(control, conf)

    pprint.pprint(conf)
    pprint.pprint(control)

    return control, control_model, control_token, conf
