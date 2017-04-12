__author__ = 'joon'

import sys

sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
sys.path.insert(0, 'ResearchTools')

from imports.basic_modules import *
from imports.ResearchTools import *


def config_test(control, conf, EXP_PHASE):
    assert (EXP_PHASE == 'saliency-test')

    defaults = dict(
        net='DeepLabv2_ResNet',
        dataset='MSRA',
        datatype='NP',
        test_datatype='Segmentation',
    )

    control_token = control.copy()
    for ky in defaults:
        if control_token[ky] == defaults[ky]:
            control_token.pop(ky)

    conf['EXP_PHASE'] = EXP_PHASE

    conf['input_padded_size'] = 321

    pprint.pprint(conf)
    pprint.pprint(control)

    return control, control_token, conf
