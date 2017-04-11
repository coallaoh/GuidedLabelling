__author__ = 'joon'


def debug_get_diff(net, name):
    return net.blobs[name].diff.copy()


def debug_get_data(net, name):
    return net.blobs[name].data.copy()
