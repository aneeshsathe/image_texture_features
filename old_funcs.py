
def an_get_img_prop(in_array=None, file_list=None, lbl_name = None, img_write_path=None):


    if file_list is not None:
        img_coll = imread_collection(file_list)
    elif in_array is not None:
        img_coll = in_array
    else:
        print('Invalid input array or file list')
        return None

    pro_img_prop_list = []
    im_count = 0
    for im_num, image in enumerate(img_coll):
        if lbl_name is None:
            sav_lbl = 'No_Label'
        else:
            sav_lbl = lbl_name[im_num]
    #        print(sav_lbl)
        label_image = an_thresh_lbl_img(image)

        img_prop_list = []

        for region_prop in regionprops(label_image, image):

            # skip small images
            if region_prop.area < 1500 or region_prop.max_intensity < 0.002:
                continue
            minr, minc, maxr, maxc = region_prop.bbox
    #            crop_im = image[minr-3:maxr+3,minc-3:maxc+3].copy()
            crop_im = image[minr:maxr, minc:maxc].copy()
            crop_glcm_prop = an_glcm(crop_im)
            crop_har_feat = mahotas.features.haralick(mahotas.stretch(crop_im), return_mean=True)
            crop_zernike_feat = mahotas.features.zernike_moments(crop_im, 5)
            local_binary_pat = an_lbp_mahotas(crop_im)
            thresh_adj_stats = an_tas_mahotas(crop_im)

            eig_val_ratio= region_prop.inertia_tensor_eigvals[1]/region_prop.inertia_tensor_eigvals[0]

            setattr(region_prop, 'glcm', crop_glcm_prop)
            setattr(region_prop, 'inertia_tensor_EV_ratio', eig_val_ratio )
            setattr(region_prop, 'crop_har_feat', crop_har_feat)
            setattr(region_prop, 'crop_zernike_feat',  crop_zernike_feat)
            setattr(region_prop, 'local_binary_pat',  local_binary_pat)
            setattr(region_prop, 'thresh_adj_stats',  thresh_adj_stats)
            setattr(region_prop, 'cell_type',  sav_lbl)

            region_dict = getregiondict(region_prop)
            img_prop_list.append(region_dict)

            im_count +=1
            if img_write_path is not None:
                crop_im = an_pad_im(region_prop.image*crop_im)
                an_sav_im(img_write_path, crop_im, sav_lbl, im_count)

        for prop in img_prop_list:
            pro_img_prop_list.append(prop)

    return pro_img_prop_list


# REF: http://earlglynn.github.io/kaggle-plankton/Plankton%20skimage%20region%20properties.html
def getregiondict(in_region):
    """
    Takes regionprops output and converts to dictionary
    :param in_region:
    :return: region_dict
    """
    region_dict = {'cell_type': in_region.cell_type,
                   'centroid_row': in_region.centroid[0],  # 0D:  location
                   'centroid_col': in_region.centroid[1],

                   'diameter_equivalent': in_region.equivalent_diameter,  # 1D
                   'length_minor_axis': in_region.minor_axis_length,
                   'length_major_axis': in_region.major_axis_length,
                   'ratio_eccentricity': in_region.eccentricity,
                   'perimeter': in_region.perimeter,

                   'orientation': in_region.orientation,  # ranges from -pi/2 to pi/2
                   'area': in_region.area,  # 2D
                   'area_convex': in_region.convex_area,
                   'area_filled': in_region.filled_area,
                   'box_min_row': in_region.bbox[0],
                   'box_max_row': in_region.bbox[2],
                   'box_min_col': in_region.bbox[1],
                   'box_max_col': in_region.bbox[3],
                   'ratio_extent': in_region.extent,
                   'ratio_solidity': in_region.solidity,

                   'inertia_tensor_eigenvalue1': in_region.inertia_tensor_eigvals[0],
                   'inertia_tensor_eigenvalue2': in_region.inertia_tensor_eigvals[1],
                   'inertia_tensor_EV_ratio'   : in_region.inertia_tensor_EV_ratio,

                   # 'moments_hu1': in_region.moments_hu[0],
                   # 'moments_hu2': in_region.moments_hu[1],
                   # 'moments_hu3': in_region.moments_hu[2],
                   # 'moments_hu4': in_region.moments_hu[3],
                   # 'moments_hu5': in_region.moments_hu[4],
                   # 'moments_hu6': in_region.moments_hu[5],
                   # 'moments_hu7': in_region.moments_hu[6],

                   'glcm_dissimilarity': in_region.glcm[0],
                   'glcm_correlation': in_region.glcm[1],
                   'glcm_contrast': in_region.glcm[2],
                   'glcm_homogeneity': in_region.glcm[3],
                   'glcm_asm': in_region.glcm[4],
                   'glcm_energy': in_region.glcm[5],


                   'euler_number': in_region.euler_number,  # miscellaneous

                   'countCoords': len(in_region.coords)}  # eventually grab these coordinates?
    an_fill_dict(region_dict, 'haralick_', in_region.crop_har_feat)
    an_fill_dict(region_dict, 'zernike_', in_region.crop_zernike_feat)
    an_fill_dict(region_dict, 'hu_', in_region.moments_hu) # translation, scale and rotation invariant
    an_fill_dict(region_dict, 'local_binary_pat_', in_region.local_binary_pat)
    an_fill_dict(region_dict, 'thresh_adj_stats_', in_region.thresh_adj_stats)
    return region_dict


def an_fill_dict(in_dict,  in_feats, in_name='F', use_dict_len=True):
    """

    :param in_dict: Dictionary to be used to add features
    :param in_feats: list of features
    :param in_name: name of features, multiple features will be added in more columns
    :param use_dict_len: Default True. to use in_name must be set to False. If True Feature is names as 'Fxx'
    :return:
    """

    if use_dict_len:
        dict_len = len(in_dict)
        in_name = 'F'
        if in_feats.size == 1:
            in_dict[in_name + str(dict_len + 1).zfill(3)] = in_feats
        else:
            for i in range(0, in_feats.size):
                dict_len = len(in_dict)
                in_dict[in_name + str(dict_len + 1).zfill(3)] = in_feats[i]

    else:
        if in_feats.size == 1:
            in_dict[in_name] = in_feats
        else:
            for i in range(0, in_feats.size):
                in_dict[in_name + str(i + 1).zfill(3)] = in_feats[i]
    return in_dict


def an_glcm(in_img):
    """
    Returns GLCM properties of image from scikit image
    Reference: http://www.fp.ucalgary.ca/mhallbey/tutorial.htm
    :param in_img:
    :return: [dissim, corr, cont, homogen, asm, ener]
    """
    glcm = greycomatrix(img_as_ubyte(in_img), [15], [0], symmetric=True, normed=True)
    dissim = greycoprops(glcm, 'dissimilarity')[0, 0]
    corr = greycoprops(glcm, 'correlation')[0, 0]
    cont = greycoprops(glcm, 'contrast')[0, 0]
    homogen = greycoprops(glcm, 'homogeneity')[0, 0]
    asm = greycoprops(glcm, 'ASM')[0, 0]
    ener = greycoprops(glcm, 'energy')[0, 0]
    return [dissim, corr, cont, homogen, asm, ener]


def an_tas_mahotas(tas_image, if_pf=True):
    """Compute parameter free Threshold Adjacency Statistics"""
    if if_pf:
        out_tas = mahotas.features.pftas(img_as_uint(tas_image, force_copy=True))
    else:
        out_tas = mahotas.features.tas(img_as_uint(tas_image, force_copy=True))
    return out_tas


i