"""an_get_img_prop extracts image texture props using functions from other packages """
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.morphology import disk
from skimage.morphology import opening
from skimage.io import imread_collection
from skimage.io import imsave
from skimage.measure import regionprops


from skimage import img_as_uint
from skimage.util import pad
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import os
import pandas as pd
from an_img_baseprops import an_base_props, an_stat_props
from an_glcm_props import glcm_props
from an_glrlm_props import glrlm_props
from an_spectral_props import an_get_spec_feats
from an_spatial_props import an_laws_feats
from an_spatial_diversity_props import an_spa_div
from an_lbp_tas_zernike import an_lbp_props, an_tas_props, an_zernike_props


def an_get_img_prop(in_array=None, img_idx=None, pool_num=0, file_list=None, lbl_name=None, img_write_path=None):

    if file_list is not None:
        img_coll = imread_collection(file_list)
    elif in_array is not None:
        img_coll = in_array
    else:
        print('Invalid input array or file list')
        return None

    feat_frame = pd.DataFrame([])
    row_count = 0
    im_count = 0

    for im_num, image in enumerate(img_coll):
        if np.std(image) > 20:  # For empty images
            if lbl_name is None:
                sav_lbl = 'No_Label'
            else:
                sav_lbl = lbl_name[img_idx[im_num]]
        #        print(sav_lbl)
            label_image = an_thresh_lbl_img(image)

            print('Image ' + str(im_num) + ' of ' + str(len(img_coll))+' Proc num: '+str(pool_num))

            for region_prop in regionprops(label_image, image):

                # skip small images
                if region_prop.area < 1500 or region_prop.max_intensity < 0.02:
                    continue
                minr, minc, maxr, maxc = region_prop.bbox
                crop_im = image[minr:maxr, minc:maxc].copy()
                bg_zero_im = crop_im * region_prop.image
                feat_frame.loc[row_count, 'CellType'] = sav_lbl
                feat_frame = an_base_props(feat_frame, row_count, region_prop, crop_im)
                feat_frame = an_stat_props(feat_frame, row_count, region_prop, crop_im)
                feat_frame = glcm_props(feat_frame, row_count, crop_im)
                feat_frame = glrlm_props(feat_frame, row_count, crop_im)
                feat_frame = an_get_spec_feats(feat_frame, row_count, crop_im)
                feat_frame = an_laws_feats(feat_frame, row_count, crop_im)
                feat_frame = an_spa_div(feat_frame, row_count, crop_im)
                feat_frame = an_lbp_props(feat_frame, row_count, bg_zero_im)
                feat_frame = an_tas_props(feat_frame, row_count, crop_im)
                feat_frame = an_zernike_props(feat_frame, row_count, bg_zero_im)
                row_count += 1

                im_count += 1
                if img_write_path is not None:
                    crop_im = an_pad_im(region_prop.image*crop_im)
                    an_sav_im(img_write_path, crop_im, sav_lbl, pool_num, im_count)

        feat_frame.fillna(np.finfo(float).eps, inplace=True)
    return feat_frame


def normalized(arr, axis=-1, order=2):
    # l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    # l2[l2==0] = 1
    # return a / np.expand_dims(l2, axis)
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[..., i].min()
        maxval = arr[..., i].max()
        if minval != maxval:
            arr[..., i] -= minval
            arr[..., i] *= (255.0 / (maxval - minval))
    return arr



def an_thresh_lbl_img(image):
    """
    Takes raw image, thresholds it, clears borders and returns labelled image
    :param image:
    :return: label_image
    """
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    # cleared = bw.copy()
    # clear_border(cleared)
    # label_image = label(bw)
    # borders = np.logical_xor(bw, cleared)
    # label_image[borders] = -1
    distance = ndi.distance_transform_edt(bw)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((35, 35)),
                                labels=bw)
    markers = ndi.label(local_maxi)[0]
    label_image = watershed(-distance, markers, mask=bw)
    # remove small objects
    selem = disk(6)
    label_image = opening(label_image, selem)

    return label_image

    # for count, a in enumerate(in_dat):
    #     if len(col_name) > 1:
    #         in_f[row_num, col_name[count]+str(count+1)] = a
    #     else:
    #         in_f[row_num, col_name[0]+str(count+1)] = a
    #
    # return in_f


def an_sav_im(img_write_path, in_img, in_lbl, pool_num, im_num):
    fold_path = os.path.join(img_write_path, in_lbl)
    if not os.path.exists(fold_path):
        os.mkdir(fold_path)
    file_name = os.path.join(fold_path, in_lbl+'_'+str(pool_num)+'_'+str(im_num).zfill(4)+'.png')
    imsave(file_name, img_as_uint(in_img))


def an_pad_im(in_im, img_size=(256, 256)):  # (128,128)):
    in_shape = in_im.shape
    x_size, y_size = [(abs((im-shp)/2), abs(im-((im-shp)/2)-shp)) for im, shp in zip(img_size, in_shape)]
    pad_im = pad(in_im, (x_size, y_size), 'constant', constant_values=0)
    return pad_im
