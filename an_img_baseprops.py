from skimage.exposure import histogram
import numpy as np
import scipy
import mahotas
from helper_func import an_df_filler


def an_base_props(feat_frame, row_count, region_prop, crop_im):
    out_props = [region_prop.area, region_prop.eccentricity,
                 region_prop.euler_number, region_prop.equivalent_diameter,
                 region_prop.extent, region_prop.major_axis_length,
                 region_prop.minor_axis_length, region_prop.max_intensity,
                 region_prop.mean_intensity, region_prop.min_intensity,
                 region_prop.orientation, region_prop.perimeter,
                 region_prop.solidity, mahotas.features.roundness(region_prop.image),
                 scipy.stats.kurtosis(crop_im.flatten()), scipy.stats.skew(crop_im.flatten())
                 ]
    out_lbl = ['Area', 'Eccentricity',
               'Euler_Num', 'Equi_Diam',
               'Extent', 'Maj_ax_len',
               'Min_ax_len', 'Max_int',
               'Mean_int', 'Min_int',
               'Orientation', 'Perimeter',
               'Solidity', 'Roundness',
               'Kurtosis', 'Skewness']

    [(out_props.append(mom), out_lbl.append('moments_' + str(i) + '_')) for i, mom in
     enumerate(np.array(region_prop.moments).flatten())]
    [(out_props.append(mom), out_lbl.append('wt_moments_' + str(i) + '_')) for i, mom in
     enumerate(np.array(region_prop.weighted_moments).flatten())]
    [(out_props.append(mom), out_lbl.append('wt_cen_moments_' + str(i) + '_')) for i, mom in
     enumerate(np.array(region_prop.weighted_moments_central).flatten())]
    [(out_props.append(mom), out_lbl.append('wt_norm_moments_' + str(i) + '_')) for i, mom in
     enumerate(np.array(region_prop.weighted_moments_normalized).flatten())]
    [(out_props.append(mom), out_lbl.append('hu_moments_' + str(i) + '_')) for i, mom in
     enumerate(np.array(region_prop.moments_hu).flatten())]
    [(out_props.append(mom), out_lbl.append('hu_wt_moments_' + str(i) + '_')) for i, mom in
     enumerate(np.array(region_prop.weighted_moments_hu).flatten())]

    out_lbl = ['BP_' + lbl + '_' for lbl in out_lbl]

    out_props = np.array(out_props)
    out_props[np.isnan(out_props)] = np.finfo(float).eps

    # feat_frame = an_df_filler(feat_frame, row_count, out_props, ['BaseProps'])
    feat_frame = an_df_filler(feat_frame, row_count, out_props, out_lbl)

    return feat_frame


def an_stat_props(feat_frame, row_count, region_prop, crop_im):
    pro_bin_prob, bin_cen = histogram(crop_im[region_prop.image])
    bin_prob = np.float16(pro_bin_prob) / sum(pro_bin_prob)

    out_props = [region_prop.mean_intensity,  # mean intensity
                 1.0 - (1.0 / (1.0 + np.square(np.std(region_prop.intensity_image)))),  # smoothness
                 np.std(crop_im[region_prop.image]),  # std
                 sum(bin_prob * np.log2(bin_prob)),  # entropy
                 sum(np.square(bin_prob)),  # Uniformity
                 scipy.stats.skew(np.ndarray.flatten(crop_im[region_prop.image])),  # skewness
                 ]
    out_lbl = ['Mean_int', 'Smoothness', 'StDev', 'Entropy', 'Uniformity', 'Skewness']
    out_lbl = ['STFOS_' + lbl + '_' for lbl in out_lbl]

    feat_frame = an_df_filler(feat_frame, row_count, out_props, out_lbl)
    # feat_frame = an_df_filler(feat_frame, row_count, out_props, ['STFOS'])

    return feat_frame

