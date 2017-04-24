import pydensecrf.densecrf as DenseCRF
import numpy as np
from matplotlib.pyplot import imshow as pim


def CRF(image, unary, crf_param, scale_factor=1.0, maxiter=10):
    assert (image.shape[:2] == unary.shape[:2])
    scale_factor = float(scale_factor)

    if 'seed' in crf_param:
        bi_w = 10.
        bi_x_std = 80.
        bi_r_std = 13.
    elif 'deeplab' in crf_param:
        bi_w = 4.
        bi_x_std = 121.
        bi_r_std = 5.
    elif 'custom' in crf_param:
        _custom, bi_w_, bi_x_std_, bi_r_std_ = crf_param.split('-')
        bi_w = float(bi_w_)
        bi_x_std = float(bi_x_std_)
        bi_r_std = float(bi_r_std_)
    else:
        raise NotImplementedError

    pos_w = pos_x_std = 3.

    H, W = image.shape[:2]
    nlabels = unary.shape[2]

    # initialize CRF
    # crf = DenseCRF(W, H, nlables)
    crf = DenseCRF.DenseCRF2D(W, H, nlabels)

    # set unary potentials
    # crf.set_unary_energy(-unary.ravel().astype('float32'))
    crf.setUnaryEnergy(-unary.transpose((2, 0, 1)).reshape((nlabels, -1)).copy(order='C').astype('float32'))

    # set pairwise potentials
    w1 = bi_w
    theta_alpha_1 = bi_x_std / scale_factor
    theta_alpha_2 = bi_x_std / scale_factor
    theta_beta_1 = bi_r_std
    theta_beta_2 = bi_r_std
    theta_beta_3 = bi_r_std
    w2 = pos_w
    theta_gamma_1 = pos_x_std / scale_factor
    theta_gamma_2 = pos_x_std / scale_factor

    # crf.add_pairwise_energy(w1,
    #                         theta_alpha_1,
    #                         theta_alpha_2,
    #                         theta_beta_1,
    #                         theta_beta_2,
    #                         theta_beta_3,
    #                         w2,
    #                         theta_gamma_1,
    #                         theta_gamma_2,
    #                         image.ravel().astype('ubyte'))
    crf.addPairwiseGaussian(sxy=(theta_gamma_1, theta_gamma_2), compat=w2, kernel=DenseCRF.DIAG_KERNEL,
                            normalization=DenseCRF.NORMALIZE_SYMMETRIC)
    crf.addPairwiseBilateral(sxy=(theta_alpha_1, theta_alpha_2), srgb=(theta_beta_1, theta_beta_2, theta_beta_3),
                             rgbim=image, compat=w1, kernel=DenseCRF.DIAG_KERNEL,
                             normalization=DenseCRF.NORMALIZE_SYMMETRIC)

    # run inference
    prediction = np.array(crf.inference(maxiter)).reshape((nlabels, H, W)).transpose((1, 2, 0))

    return prediction
