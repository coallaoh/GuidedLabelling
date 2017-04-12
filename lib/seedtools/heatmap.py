__author__ = 'joon'

import scipy.ndimage as nd


def heatmap2segconf(heat_maps, imshape_original, gt_cls):
    heat_maps_os = nd.zoom(heat_maps,
                           [1,
                            float(imshape_original[0]) / heat_maps.shape[1],
                            float(imshape_original[1]) / heat_maps.shape[2]],
                           order=1)
    heat_maps_norm = heat_maps_os / heat_maps_os.max(axis=1).max(axis=1).reshape((-1, 1, 1))
    confidence = heat_maps_norm.max(0)
    seg = gt_cls[heat_maps_norm.argmax(axis=0)]

    return seg, confidence
