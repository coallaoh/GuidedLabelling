__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
sys.path.insert(0, 'ResearchTools')

from imports.basic_modules import *
from imports.ResearchTools import *
from imports.libmodules import *


def config_generate(control, conf, EXP_PHASE):
    assert (EXP_PHASE == 'guide-generate')

    assert (control['g_test_dataset'] == control['s_test_dataset'])
    assert (control['test_dataset'] == control['s_test_dataset'])
    assert (control['g_test_datatype'] == control['s_test_datatype'])
    assert (control['test_datatype'] == control['s_test_datatype'])

    control_seed = subcontrol(control, 'g')
    control_saliency = subcontrol(control, 's')

    from seed.config import config_test as config_test_seed
    from saliency.config import config_test as config_test_saliency

    control_seed, _, control_seed_token, _ = config_test_seed(control_seed, conf, 'seed-test')
    control_saliency, control_saliency_token, _ = config_test_saliency(control_saliency, conf, 'saliency-test')

    defaults = dict(
        # seed
        g_init='VGG_ILSVRC_16_layers',
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
        test_dataset='voc12train_aug',
        test_datatype='Segmentation',
    )

    control_token = control.copy()
    for ky in defaults:
        if control_token[ky] == defaults[ky]:
            control_token.pop(ky)

    conf['EXP_PHASE'] = EXP_PHASE
    load_pascal_conf(control, conf)

    pprint.pprint(conf)
    pprint.pprint(control)

    return control, control_token, control_seed, control_seed_token, control_saliency, control_saliency_token, conf
