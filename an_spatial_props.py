import numpy as np
import scipy
from an_glcm_props import an_glcm_entropy
from helper_func import an_df_filler


def an_laws_feats(feat_frame, row_num, crop_im):
    pro_law_vecs = [[1, 2, 1],  # L3
                    [-1, 0, 1],  # E3
                    [1, -2, -1],  # S3
                    [1, 4, 6, 4, 1],  # L5
                    [-1, -2, 0, 2, 1],  # E5
                    [-1, 0, 2, 0, -1],  # S5
                    [-1, 2, 0, -2, -1],  # W5
                    [1, -4, 6, -4, 1],  # R5
                    [1, 6, 15, 20, 15, 6, 1],  # L7
                    [-1, -4, -5, 0, 5, 4, 1],  # E7
                    [-1, -2, 1, 4, 1, -2, -1],  # S7
                    [1, 8, 28, 56, 70, 56, 28, 8, 1],  # L9
                    [1, 4, 4, -4, -10, -4, 4, 4, 1],  # E9
                    [1, 0, -4, 0, 6, 0, -4, 0, 1],  # S9
                    [1, -4, 4, -4, -10, 4, 4, -4, 1],  # W9
                    [1, -8, 28, -56, 70, -56, 28, -8, 1]  # R9
                    ]

    pro_pro_lbl = ['L3', 'E3', 'S3', 'L5', 'E5', 'S5', 'W5', 'R5', 'L7', 'E7', 'S7', 'L9', 'E9', 'S9', 'W9', 'R9']

    law_vec_sets = [[0, 2], [3, 7], [8, 10], [11, 15]]  # define groups
    pro_laws_param = []
    out_lbl = []
    for f in law_vec_sets:
        law_vecs = np.array([np.atleast_2d(xi) for xi in pro_law_vecs[f[0]:f[1] + 1]])
        pro_lbl = [pro_pro_lbl[i] for i, j in enumerate(pro_law_vecs[f[0]:f[1] + 1])]
        for i, vec1 in enumerate(law_vecs):
            for j, vec2 in enumerate(law_vecs):
                if i == j:
                    # print(pro_[i,j])
                    law_masks = [vec1 * vec2.T]
                    TR = np.squeeze(an_laws_TEI(crop_im, law_masks))
                    TR = np.ravel(TR)
                    pro_laws_param.append(
                        [np.std(TR), scipy.stats.skew(TR), scipy.stats.kurtosis(TR), an_glcm_entropy(TR)])
                    lb = ['StDev', 'Skew', 'Kurt', 'Entropy']
                    out_lbl.append(['SF_' + pro_lbl[i] + '_' + pro_lbl[j] + '_' + l + '_' for l in lb])

                if j > i:
                    # print([i,j])
                    law_masks = [vec1 * vec2.T, vec2 * vec1.T]
                    TEI_imgs = an_laws_TEI(crop_im, law_masks)
                    TR = np.squeeze((TEI_imgs[0] + TEI_imgs[0]) / 2)
                    TR = np.ravel(TR)
                    pro_laws_param.append(
                        [np.std(TR), scipy.stats.skew(TR), scipy.stats.kurtosis(TR), an_glcm_entropy(TR)])
                    lb = ['StDev', 'Skew', 'Kurt', 'Entropy']
                    out_lbl.append(['SF_' + pro_lbl[i] + '_' + pro_lbl[j] + '_' + l + '_' for l in lb])

    laws_param = np.reshape(pro_laws_param, -1)
    out_lbl = np.reshape(out_lbl, -1)
    # print(np.array(out_lbl).shape)
    # print(laws_param.shape)
    # laws_param.shape
    feat_frame = an_df_filler(feat_frame, row_num, laws_param, out_lbl)

    return feat_frame


# Helper functions


def an_laws_TI(crop_im, law_masks):
    return np.array([np.array(scipy.signal.convolve2d(crop_im,in_mask, boundary='fill', fillvalue=0)) for in_mask in law_masks])


def an_laws_TEI(crop_im, law_masks):
    conv_imgs = an_laws_TI(crop_im, law_masks)
    corr_fac = [7, 7]
    img_shape1 = conv_imgs[0].shape[0]-corr_fac[0]
    img_shape2 = conv_imgs[0].shape[1]-corr_fac[1]

    out_TEI = np.zeros((conv_imgs.shape[0], img_shape1-corr_fac[0],img_shape2-corr_fac[0]))
    for k, in_img in enumerate(conv_imgs):
        a = np.array([np.sum(np.abs(in_img[i-7:i+8, j-7:j+8]))
                      for i in range(corr_fac[0], img_shape1) for j in range(corr_fac[1], img_shape2)])
        out_TEI[k, :, :] = np.reshape(a, [img_shape1-corr_fac[0],img_shape2-corr_fac[0]])

    return out_TEI
