import numpy as np
import scipy
from skimage import img_as_ubyte
from skimage.feature import greycomatrix, greycoprops

from helper_func import an_df_filler


def glcm_props(feat_frame, row_num, crop_im):
    in_dist = [1, 2, 4, 8, 10]
    in_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = greycomatrix(img_as_ubyte(crop_im), in_dist, in_angles, symmetric=True, normed=True)
    # Replace 0s with small value
    glcm += np.finfo(float).eps

    out_props = [an_glcm_cont(glcm),  # Contrasts, their means and ranges
                 an_glcm_dissim(glcm),  # Dissimilarity
                 an_glcm_homogen(glcm),  # Homogeneity
                 an_glcm_asm(glcm),  # Angular Second Moment
                 an_glcm_ener(glcm),  # Energy
                 an_glcm_corr(glcm),  # Correlation
                 an_iter_func_over_glcm(an_glcm_entropy, glcm),  # Entropy
                 ]

    no_of_val2return = 15  # for SVD

    out_props = np.ravel(out_props)
    out_props = np.concatenate((out_props,
                                glcm_more_feats(glcm), # ['sumAvg','SumVariance','DiffVariance','SumEntropy','DiffEntropy','IMC1','IMC2']
                                an_glcm_svd(glcm, no_of_val2return=no_of_val2return)  # SVD
                                ))
    # print(out_props.shape)
    u_ang = np.degrees(in_angles).astype(int)

    b_lbl = ['Con', 'Dsm', 'Hom', 'ASM', 'Enr', 'Cor', 'Ent', ]
    out_lbl = np.ravel([glcm_nm(a, u_ang, in_dist, lbl_type='Base') for a in b_lbl])

    b_lbl = ['muSum', 'Var', 'sumAvg', 'SumVariance', 'DiffVariance', 'SumEntropy', 'DiffEntropy', 'IMC1', 'IMC2']
    out_lbl = np.append(out_lbl, np.ravel(glcm_nm(b_lbl, u_ang, in_dist, lbl_type='More')))

    out_lbl = np.append(out_lbl,
                        np.ravel(glcm_nm(b_lbl, u_ang, in_dist, lbl_type='SVD', no_of_val2return=no_of_val2return)))
    # print(feat_frame.shape)
    # print(out_props.shape)
    # print(out_lbl.shape)

    #     feat_frame = an_df_filler(feat_frame, row_num, out_props, ['STSOS'])
    feat_frame = an_df_filler(feat_frame, row_num, out_props, out_lbl)  # ['STSOS'])
    return feat_frame


# HELPER FUNCTIONS


def an_glc_mean(in_prp):
    """
    Calculates mean of incoming property along axis 1
    :param in_prp:
    :return:
    """
    return np.sum(in_prp, axis=1) / in_prp.shape[1]


def an_glc_range(in_prp):
    """
    Calculates range of incoming property along axis 1
    :param in_prp:
    :return:
    """
    return np.max(in_prp, axis=1) - np.min(in_prp, axis=1)


def an_join_prop_mean_range(in_prop):
    """
    Joins incoming features and their mean and range values by row. Then returns as single row
    :param in_prop:
    :return:
    """
    in_mean = np.sum(in_prop, axis=1) / in_prop.shape[1]
    # in_mean = an_glc_mean(in_prop)
    in_range = np.max(in_prop, axis=1) - np.min(in_prop, axis=1)
    # in_range = an_glc_range(in_prop)
    # print(in_mean)
    # print(in_range)
    return np.ravel([np.append(in_prop[i], [in_mean[i], in_range[i]]) for i, _ in enumerate(in_prop)])


def an_iter_func_over_glcm(in_func, in_glcm):
    """
    Applies incoming function in_func to incoming glcm, in_glcm
    :param in_func:
    :param in_glcm:
    :return:
    """
    prop = [in_func(in_glcm[:, :, i, j]) for i in range(in_glcm.shape[-2]) for j in range(in_glcm.shape[-1])]
    prop = np.array(prop).reshape((in_glcm.shape[-2], in_glcm.shape[-1]))
    # print prop
    return an_join_prop_mean_range(prop)


def an_glcm_base_vals(in_glcm):
    f = np.array([in_glcm[i, j] for i in range(in_glcm.shape[-2]) for j in range(in_glcm.shape[-1])
                  if i+j in range(2, 2*in_glcm.shape[0])])
    p_xplusy = f[range(2, 2*in_glcm.shape[0])]
    p_xminusy = f[range(0, in_glcm.shape[0]-1)]
    # print(p_xplusy.shape)
    # print(p_xminusy.shape)
    return p_xplusy, p_xminusy

# FEATURE FUNCTIONS


def an_glcm_cont(glcm):
    """
    Calculates Contrast, mean and range for each glcm matrix
    :param glcm:
    :return: row in form [contrast 1.1, contrast 1.2,...mean1, range1, contrast 2.1..]
    """
    cont = greycoprops(glcm, 'contrast')  #
    # print(cont)
    return an_join_prop_mean_range(cont)


def an_glcm_dissim(glcm):
    """
    Calculates Dissimilarity of incoming glcm. Returns dissimilarity, mean, range like contrast
    :param glcm:
    :return:
    """
    dissim = greycoprops(glcm, 'dissimilarity')
    # print(dissim)
    return an_join_prop_mean_range(dissim)


def an_glcm_homogen(glcm):
    """
    Calculates homogeneity of incoming glcm
    :param glcm:
    :return:
    """
    homogen = greycoprops(glcm, 'homogeneity')
    # print(homogen)

    return an_join_prop_mean_range(homogen)


def an_glcm_asm(glcm):
    """
    Calculates Angular Second Moment
    :param glcm:
    :return:
    """
    asm = greycoprops(glcm, 'ASM')
    return an_join_prop_mean_range(asm)


def an_glcm_ener(glcm):
    """
    Calculates Energy of incoming GLCMs
    :param glcm:
    :return:
    """
    ener = greycoprops(glcm, 'energy')
    # print(ener)
    return an_join_prop_mean_range(ener)


def an_glcm_corr(glcm):
    """
    Calculates Correlation of incoming GLCMs
    :param glcm:
    :return:
    """
    corr = greycoprops(glcm, 'correlation')
    # print(corr)
    return an_join_prop_mean_range(corr)


def an_glcm_entropy(in_glcm):
    pro_ent = -np.log(in_glcm)
    pro_ent[np.isinf(pro_ent) | np.isneginf(pro_ent)] = np.finfo(float).eps
    out_ent = np.sum(in_glcm*pro_ent)
    return out_ent


def glcm_more_feats(glcm):
    """
    Calculates features with shared variables.
    :param glcm:
    :return:
    """
    out_l = []

    for count1 in range(glcm.shape[-2]):
        for count2 in range(glcm.shape[-1]):

            in_glcm = glcm[:, :, count1, count2]
            p_xplusy, p_xminusy = an_glcm_base_vals(in_glcm)

            mu_i = [in_glcm[i, :] * i for i in range(0, in_glcm.shape[0])]
            i_minus_usq = ((range(in_glcm.shape[0]) - mu_i[0]) ** 2)
            an_var = np.sum([i_minus_usq[i] * in_glcm[i, :] for i in range(in_glcm.shape[0])])

            sumAvg = np.sum(range(2, 2*in_glcm.shape[0])*p_xplusy)
            SumVariance = np.sum((np.array(range(2,2*in_glcm.shape[0])-sumAvg)**2)*p_xplusy)
            DiffVariance = np.var(p_xminusy)
            SumEntropy = -np.sum(p_xplusy*np.log(p_xplusy))
            DiffEntropy = np.sum(-p_xminusy*np.log(p_xminusy))

            px_i = np.sum(in_glcm, axis=1)
            py_j = np.sum(in_glcm, axis=0)
            HX = scipy.stats.entropy(px_i)
            HY = scipy.stats.entropy(py_j)
            HXY  = -np.sum([in_glcm[i, j]*np.log(in_glcm[i, j]) for i in range(in_glcm.shape[0]) for j in range(in_glcm.shape[1])])
            HXY1 = -np.sum([in_glcm[i, j]*np.log(px_i[i]*py_j[j]) for i in range(in_glcm.shape[0]) for j in range(in_glcm.shape[1])])
            HXY2 = -np.sum([px_i[i]*py_j[j]*np.log(px_i[i]*py_j[j]) for i in range(in_glcm.shape[0]) for j in range(in_glcm.shape[1])])
            IMC1 = HXY-HXY1/np.max([HX, HY])
            IMC2 = np.sqrt(1-np.exp(-2*(HXY2-HXY)))
#             print IMC2

            out_l.append([np.sum(mu_i),
                          an_var,
                          sumAvg,
                          SumVariance,
                          DiffVariance,
                          SumEntropy,
                          DiffEntropy,
                          IMC1,
                          IMC2
                          ])

    return np.ravel(out_l)


def an_glcm_svd(in_glcm, no_of_val2return=15):
    """
    Calculates the single value decompositions of the incoming glcm and returns the diagonals of
    the most informative values(15 by default).
    :param in_glcm:
    :param no_of_val2return:
    :return:
    """
    pro_out_svd = []
    for i in range(0, in_glcm.shape[3]):
        u, s, v = np.linalg.svd(in_glcm[:, :, :, i])
        s_sorted = -np.sort(-s, axis=0)
        pro_out_svd.append(np.transpose(s_sorted[0:no_of_val2return]))
        # pyplot.plot(s_sorted[0:no_of_val2return])

    out_svd = np.ravel(pro_out_svd)

    return out_svd


def glcm_nm(in_nm,in_angles, in_dist, lbl_type, no_of_val2return=0):

    a_lbl = [['GLCM_t_'+str(a) +'_d_'+str(d) for a in in_angles] for d in in_dist ]

    if (lbl_type=='Base'):
        b_lbl = [np.append(a, ['GLCM_d_'+str(in_dist[i])+'_avg','GLCM_d_'+str(in_dist[i])+'_rng']) for i,a in enumerate(a_lbl) ]
        return [b+'_'+in_nm+'_' for b in np.ravel(b_lbl)]
    elif (lbl_type=='More'):
        return [b+'_'+c+'_' for b in np.ravel(a_lbl) for c in in_nm]
    elif (lbl_type=='SVD'):
        s_lbl = ['SVD'+str(s) for s in range(1, no_of_val2return+1)]
        return [b+'_'+c+'_' for b in np.ravel(a_lbl) for c in s_lbl]
