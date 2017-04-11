__author__ = 'joon'


def disp_net(net):
    print ('\n=====\nBLOBS\n=====\n')
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)

    print ('\n=====\nLAYERS\n=====\n')
    for layer_name, param in net.params.iteritems():
        if len(param) > 1:
            print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
        else:
            print layer_name + '\t' + str(param[0].data.shape)
