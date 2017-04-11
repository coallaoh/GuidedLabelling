__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
sys.path.insert(0, 'ResearchTools')

from imports.basic_modules import *
from imports.ResearchTools import *


def config_train(control, conf, EXP_PHASE):
    assert (EXP_PHASE == 'seed-train')

    default_control = dict(
        init='VGG_ILSVRC_16_layers',
        dataset='voc12train_aug',
        datatype='Segmentation',
        base_lr=0.001,
        batch_size=15,
        balbatch='clsbal',
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
    if 'voc' in control['dataset']:
        conf['nposcls'] = 20
    else:
        raise NotImplementedError

    pprint.pprint(conf)
    pprint.pprint(control)

    return control, control_token, conf
