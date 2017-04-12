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


def config_test(control, conf, EXP_PHASE):
    assert (EXP_PHASE == 'seed-test')

    defaults = dict(
        init='VGG_ILSVRC_16_layers',
        dataset='voc12train_aug',
        datatype='Segmentation',
        base_lr=0.001,
        batch_size=15,
        balbatch='clsbal',
        test_iter=8000,
        test_datatype='Segmentation',
        test_ranking='none',
        test_gtcls='use',
    )

    defaults_model = dict(
        init='VGG_ILSVRC_16_layers',
        dataset='voc12train_aug',
        datatype='Segmentation',
        base_lr=0.001,
        batch_size=15,
        balbatch='clsbal',
    )

    exclude_from_model = [
        'test_iter',
        'test_dataset',
        'test_datatype',
        'test_ranking',
        'test_gtcls',
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
    if 'voc' in control['dataset']:
        conf['nposcls'] = 20
    else:
        raise NotImplementedError

    if control['net'] == 'GAP-LowRes':
        conf['out_size'] = 21
        conf['blobname_map'] = 'relu6_gap'
        conf['paramname_weight'] = 'score'
    elif control['net'] == 'GAP-HighRes':
        conf['out_size'] = 41
        conf['blobname_map'] = 'relu7_gap'
        conf['paramname_weight'] = 'score'
    elif control['net'] == 'GAP-ROI':
        conf['out_size'] = 41
        raise NotImplementedError
    elif control['net'] == 'GAP-DeepLab':
        conf['out_size'] = 41
        conf['blobname_map'] = 'relu7'
        conf['paramname_weight'] = 'score'
    else:
        raise NotImplementedError

    pprint.pprint(conf)
    pprint.pprint(control)

    return control, control_model, control_token, conf


def config_eval(control, conf, EXP_PHASE):
    assert (EXP_PHASE == 'seed-eval')

    defaults = dict(
        init='VGG_ILSVRC_16_layers',
        dataset='voc12train_aug',
        datatype='Segmentation',
        base_lr=0.001,
        batch_size=15,
        balbatch='clsbal',
        test_iter=8000,
        test_datatype='Segmentation',
        test_ranking='none',
        test_gtcls='use',
    )

    control_token = control.copy()
    for ky in defaults:
        if control_token[ky] == defaults[ky]:
            control_token.pop(ky)

    conf['EXP_PHASE'] = EXP_PHASE
    conf['nclasses'] = 20
    conf['year'] = '20' + control['test_dataset'][3:5]
    conf['testset'] = control['test_dataset'][5:]
    conf['imgsetpath'] = osp.join(conf['pascalroot'], 'VOC' + conf['year'], 'ImageSets', 'Segmentation', 'list',
                                  '%s.txt')
    if '_aug' in conf['testset']:
        conf['clsimgpath'] = osp.join(conf['pascalroot'], 'VOC' + conf['year'], 'SegmentationClassAug', '%s.png')
    else:
        conf['clsimgpath'] = osp.join(conf['pascalroot'], 'VOC' + conf['year'], 'SegmentationClass', '%s.png')

    conf['clsgt'] = osp.join(conf['pascalroot'], 'VOC' + conf['year'], 'ImageSets', 'Main',
                             '%s_' + conf['testset'] + '.txt')

    conf['thresrange'] = np.hstack([
        np.linspace(-10, -2, 9),
        np.linspace(-1, -0.01, 100),
        np.linspace(0, 0.99, 100),
    ])
    conf['threstype'] = 'generic'

    conf['plotconf'] = dict(
        label=control['net'],
        marker='x',
        colour='#000000',
        line='-',
        order=1,
        ms=5,
        fs_label=32,
    )

    pprint.pprint(conf)
    pprint.pprint(control)

    return control, control_token, conf


def config_eval_cls(control, conf, EXP_PHASE):
    assert (EXP_PHASE == 'seed-eval-cls')

    defaults = dict(
        init='VGG_ILSVRC_16_layers',
        dataset='voc12train_aug',
        datatype='Segmentation',
        base_lr=0.001,
        batch_size=15,
        balbatch='clsbal',
        test_iter=8000,
        test_datatype='Segmentation',
        test_ranking='none',
        test_gtcls='use',
    )

    control_token = control.copy()
    for ky in defaults:
        if control_token[ky] == defaults[ky]:
            control_token.pop(ky)

    conf['EXP_PHASE'] = EXP_PHASE
    conf['year'] = '20' + control['test_dataset'][3:5]
    conf['testset'] = control['test_dataset'][5:]
    conf['clsgt'] = osp.join(conf['pascalroot'], 'VOC' + conf['year'], 'ImageSets', 'Main',
                             '%s_' + conf['testset'] + '.txt')

    pprint.pprint(conf)
    pprint.pprint(control)

    return control, control_token, conf
