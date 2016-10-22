from math import *
import numpy as np
from skimage import img_as_ubyte
from helper_func import an_df_filler


def glrlm_props(feat_frame, row_num, crop_im, pix_dist=[2], pix_angle=[0, 45, 90, 135]):
    """
    Calculates Gray Level Run Length features. Default distance is 2 and angles 0,45,90,135.
    :param feat_frame:
    :param crop_im:
    :param pix_dist:
    :param pix_angle:
    :return:
    """

    # pix_dist = [1, 2, 4, 8]
    # pix_angle = [0, 45, 90, 135]

    out_glrlm = [get_glrlm_props(crop_im, dist, ang) for dist in pix_dist for ang in pix_angle]
    out_glrlm = np.ravel(out_glrlm)

    out_l = ['SRE', 'LRE', 'GLN', 'RLN', 'RP','LGLRE','HGLRE', 'SRLGLE', 'SRHGLE', 'LRLGLE', 'LRHGLE']
    out_lbl = ['STHOS_r_'+str(dist)+'_a_'+str(ang)+'_'+l+'_' for dist in pix_dist for ang in pix_angle for l in out_l]
    # print(out_lbl)
    # print(np.ravel(out_glrlm).shape)

    feat_frame = an_df_filler(feat_frame, row_num, out_glrlm, out_lbl)


    return feat_frame

# Helper Functions


def an_get_pts_at_dist(in_row, in_col, in_radius, in_angle):
    out_row = sin(radians(in_angle))*in_radius
    out_col = cos(radians(in_angle))*in_radius

    if out_row < 0:
        out_row = floor(out_row)
    else:
        out_row = ceil(out_row)
    if out_col < 0:
        out_col = floor(out_col)
    else:
        out_col = ceil(out_col)

    out_row = int(in_row+out_row)
    out_col = int(in_col+out_col)
    return out_row, out_col


# Prop Functions


def get_glrlm_props(crop_im, pix_dist, pix_angle):
    img = img_as_ubyte(crop_im)
    # print(img.shape)
    max_int = np.max(img)
    min_int = np.min(img)
    gl = (max_int-min_int)+1

    # imshow(skimage.exposure.equalize_hist(img))
    mc = 0
    # pix_dist = 2
    # pix_angle = 45
    count = 1
    c = 0
    col = 0
    maxcount = np.zeros((img.shape[0] * img.shape[1]))
    grl = np.zeros((max_int, max([len(np.diagonal(img)), img.shape[0], img.shape[1]])))

    for row_count in range(0, img.shape[0] - pix_dist):

        for col_count in range(0, img.shape[1] - pix_dist):

            mc += 1
            ref_pix = img[row_count, col_count]
            test_pix = img[an_get_pts_at_dist(row_count - 1, col_count - 1, pix_dist, pix_angle)]

            if ref_pix == test_pix & ref_pix != 0:
                count += 1
                c = count
                col = count
                maxcount[mc] = count
            else:
                grl[ref_pix - 1, c] += 1;
                col = 1
                count = 1
                c = 0

        grl[test_pix - 1, col - 1] += 1
        count = 1
        c = 1

        # print(grl)
    # print(grl.shape)
    np.max(maxcount)

    g_row = np.sum(grl, axis=0)
    r_col = np.sum(grl, axis=1)
    s = np.sum(grl)
    j_sq = (np.array(range(grl.shape[1]))+1)**2
    i_sq = (np.array(range(grl.shape[0]))+1)**2

    # props
    SRE = np.sum((g_row/s)/j_sq)
    LRE = np.sum((g_row/s)*j_sq)
    GLN = np.sum(np.sum(grl, axis=1)**2)/np.sum(grl)
    RLN = np.sum(np.sum(grl, axis=0)**2)/np.sum(np.ravel(grl))
    RP = np.sum(np.ravel(grl))/(img.size/pix_dist)
    LGLRE = np.sum((r_col/s)/i_sq)
    HGLRE = np.sum((r_col/s)*i_sq)
    SRLGLE = np.sum([grl[i-1, j-1]/((i**2)*(j**2)) for i in range(1, grl.shape[0]+1) for j in range(1, grl.shape[1]+1)])/s
    SRHGLE = np.sum([grl[i-1, j-1]*(i**2)/(j**2) for i in range(1, grl.shape[0]+1) for j in range(1, grl.shape[1]+1)])/s
    LRLGLE = np.sum([grl[i-1, j-1]*(j**2)/(i**2) for i in range(1, grl.shape[0]+1) for j in range(1, grl.shape[1]+1)])/s
    LRHGLE = np.sum([grl[i-1, j-1]*(j**2)*(i**2) for i in range(1, grl.shape[0]+1) for j in range(1, grl.shape[1]+1)])/s

    out_l = [SRE, LRE, GLN, RLN, RP,LGLRE, HGLRE, SRLGLE, SRHGLE, LRLGLE, LRHGLE]

    return out_l
