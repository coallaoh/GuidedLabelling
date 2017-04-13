__author__ = 'joon'

import os


def save_to_caffemodel(net, learnedmodel, OVERRIDECACHE):
    if not OVERRIDECACHE:
        assert (not os.path.isfile(learnedmodel))
    else:
        if os.path.isfile(learnedmodel):
            print ('WARNING: OVERRIDING EXISTING RESULT FILE')
    net.save(str(learnedmodel))
    print ('results saved to %s' % learnedmodel)
