from skimage.draw import circle
import numpy as np
import math
from scipy import ndimage as ndi
from scipy.stats import kurtosis
from skimage.filters import gabor_kernel
from scipy.signal import correlate2d
from helper_func import an_df_filler


def an_get_spec_feats(feat_frame, row_num, crop_im):
    out_spec1, lbl1 = an_fourier_feats(crop_im)
    out_spec1[np.isnan(out_spec1)] = 0
    feat_frame = an_df_filler(feat_frame, row_num, out_spec1, lbl1)

    out_spec2, lbl2 = an_gabor_feats(crop_im)
    out_spec2[np.isnan(out_spec2)] = 0
    feat_frame = an_df_filler(feat_frame, row_num, out_spec2, lbl2)

    out_spec3, lbl3 = an_gen_corr_feat(crop_im)
    out_spec3[np.isnan(out_spec3)] = 0
    feat_frame = an_df_filler(feat_frame, row_num, out_spec3, lbl3)


    return feat_frame



# Helper Functions


def an_gen_ring(radii, psd2D):
    img_cen = np.array(psd2D.shape)/2
    ring_masks = np.zeros((len(radii), psd2D.shape[0], psd2D.shape[1]))
    for count, rad in enumerate(radii):
        idx_img = np.zeros(psd2D.shape)
        rr1, cc1 = circle(img_cen[0], img_cen[1], rad, shape=psd2D.shape)
        rr2, cc2 = circle(img_cen[0], img_cen[1], rad-5, shape=psd2D.shape)
        idx_img[rr1, cc1] = 1
        idx_img[rr2, cc2] = 0

        ring_masks[count, :, :] = idx_img

    return ring_masks


def an_gen_wedge(in_theta, psd2D):
    img_cen = np.array(psd2D.shape)/2

    idx_img = np.zeros((len(in_theta), psd2D.shape[0], psd2D.shape[1]))

    for count, t in enumerate(in_theta):
        theta1 = t - (45/2)
        theta2 = t + (45/2)+1

        # math.tan(math.radians(theta1))
#         print([theta1,theta2])
        rw = []
        cw = []

        i_range = range(img_cen[0]-2*img_cen[0], img_cen[0])
        j_range = range(img_cen[1]-2*img_cen[1], img_cen[1])

    #     print(i_range)
        for i in np.array(i_range):
            for j in np.array(j_range):
                co_rat = math.degrees(math.atan2(j, i))
                if theta1 <= co_rat < theta2:
                    rw.append(i+img_cen[0])
                    cw.append(j+img_cen[1])

        idx_img[count, rw, cw] = 1

    return idx_img


# Prop Functions

def an_fourier_feats(crop_im):
    fft_im = np.fft.fftshift(np.fft.fft2(crop_im))

    # Calculate a 2D power spectrum
    psd2D = np.abs(fft_im) ** 2
    img_cen = np.array(psd2D.shape) / 2

    # radii = range(5, int(np.sqrt(img_cen[0] ** 2 + img_cen[1] ** 2) + 5), 5)
    radii = np.linspace(5, int(np.sqrt(img_cen[0] ** 2 + img_cen[1] ** 2) + 5), 5)
    in_theta = [0, 45, 90, 135]


    ring_imgs = an_gen_ring(radii, psd2D)
    wedge_imgs = an_gen_wedge(in_theta, psd2D)

    rw_int = [('SPFPS_R' + str(i) + '_T' + str(in_theta[j]) + '_', np.logical_and(im1, im2)) for i, im1 in
              enumerate(ring_imgs) for j, im2 in enumerate(wedge_imgs)]

    # rw_int = [('SPFPS_R' + str(radii[i]) + '_T' + str(in_theta[j]) + '_', np.logical_and(im1, im2)) for i, im1 in
    #           enumerate(ring_imgs) for j, im2 in enumerate(wedge_imgs)]

    #     print([rw_int[i][1] for i,im in enumerate(rw_int)])
    rw_sums = np.ravel([np.sum(psd2D[rw_int[i][1]]) for i, im in enumerate(rw_int)])
    out_col_nm = [rw_int[i][0] for i, im in enumerate(rw_int)]
    # len(rw_sums)

    return rw_sums, out_col_nm


def compute_feats(image, kernels, out_col_nm):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    out_nm = []
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
        out_nm.append(out_col_nm[k] + 'mean')
        out_nm.append(out_col_nm[k] + 'var')
    return np.ravel(feats), np.ravel(out_nm)


def an_gabor_feats(crop_im):
    # prepare filter bank kernels
    kernels = []
    out_col_nm = []
    # for theta in range(4):
    for theta in [22.5, 45, 67.5, 90, 112.5, 135, 157.5]:
        #     theta = theta / 4. * np.pi
        theta1 = np.radians(theta)
        for sigma in ([1, 2, 3]):  # sigma is scale
            for frequency in ([0.05, 0.25, 0.4]):
                kernel = np.real(gabor_kernel(frequency, theta=theta1,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
                out_col_nm.append('SPGWT_T' + str(int(theta * 10)) + '_S' + str(sigma) + '_')

    feats, out_col_nm = compute_feats(crop_im, kernels, out_col_nm)
    #     print(feats.shape)
    #     print(feats)
    gab_feats = np.ravel(feats)

    #     print(gab_feats)

    return gab_feats, out_col_nm


def an_gen_corr_feat(crop_im):
    corr_im = correlate2d(crop_im, crop_im, mode='full', boundary='symm')
    img_cen = np.array(corr_im.shape) / 2
    # radii = range(5, int(np.sqrt(img_cen[0] ** 2 + img_cen[1] ** 2) + 5), 5)
    radii = np.linspace(5, int(np.sqrt(img_cen[0] ** 2 + img_cen[1] ** 2) + 5), 5)
    corr_ring_masks = an_gen_ring(radii, corr_im) > 0

    pro_out_feat = [([np.sum(corr_im[im]), np.mean(corr_im[im]),
                      np.median(corr_im[im]), np.max(corr_im[im]),
                      np.min(corr_im[im]),
                      np.std(corr_im[im]), kurtosis(corr_im[im])],
                     ['SPCF_' + str(i) + '_sum_', 'SPCF_' + str(i) + '_mean_',
                      'SPCF_' + str(i) + '_med_', 'SPCF_' + str(i) + '_max_',
                      'SPCF_' + str(i) + '_min_',
                      'SPCF_' + str(i) + '_std_', 'SPCF_' + str(i) + '_kur_']) for i, im in enumerate(corr_ring_masks)]

    out_feat = np.reshape([f[0] for f in pro_out_feat], -1)
    out_lbl = np.reshape([f[1] for f in pro_out_feat], -1)
    return out_feat, out_lbl
