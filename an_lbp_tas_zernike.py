import numpy as np
import skimage
import mahotas
from helper_func import an_df_filler


def an_lbp_props(feat_frame, row_num, bg_zero_im):
    radii = [1, 3, 5, 10]
    no_of_points = 10

    out_lbp = np.array([an_lbp_mahotas(bg_zero_im, radius, no_of_points) for radius in radii])
    out_lbl = ['LBP_r' + str(r) + '_pt_' + str(pt) + '_' for r in range(out_lbp.shape[0]) for pt in
               range(out_lbp.shape[1])]
    out_lbp = out_lbp.flatten()

    feat_frame = an_df_filler(feat_frame, row_num, out_lbp, out_lbl)

    return feat_frame


def an_tas_props(feat_frame, row_num, crop_im):
    an_tas = mahotas.features.pftas(skimage.img_as_uint(crop_im))

    feat_frame = an_df_filler(feat_frame, row_num, an_tas, ['TAS'])

    return feat_frame


def an_zernike_props(feat_frame, row_num, bg_zero_im):
    radii = [1, 2, 3, 5, 10]
    out_zernike = np.array([an_zernike_mahotas(bg_zero_im, radius) for radius in radii])
    # print(out_zernike.shape)
    out_lbl = ['ZER_r'+str(r)+'_pt_'+str(pt)+'_' for r in range(out_zernike.shape[0]) for pt in range(out_zernike.shape[1])]
    # print(out_lbl)
    out_zernike = out_zernike.flatten()

    feat_frame = an_df_filler(feat_frame, row_num, out_zernike, out_lbl)
    return feat_frame


def an_lbp_mahotas(lbp_image, radius=5, n_points=6):
    # n_points = 3 * radius
    return mahotas.features.lbp(lbp_image, radius, n_points, ignore_zeros=True)


def an_zernike_mahotas(zer_image, in_radius):
    """Compute parameter free Threshold Adjacency Statistics"""
    return mahotas.features.zernike_moments(zer_image, radius=in_radius)