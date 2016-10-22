from skimage import img_as_ubyte
import numpy as np
import math
from helper_func import an_df_filler


def an_spa_div(feat_frame, row_num, crop_im):
    img = img_as_ubyte(crop_im)
    Np = img.size

    s = np.unique(img)
    # print(s)
    S = s.size
    # print(S)

    n_i, bin_edges = np.histogram(img, bins=256)
    # Replace 0s with small values
    n_i = n_i.astype(float) + np.finfo(float).eps
    # print(n_i)

    p_i = n_i / float(Np)
    # Replace 0s with small values
    p_i += + np.finfo(float).eps
    # print(p_i)

    H = -np.sum(p_i * np.log(p_i))  # Shannon Wiener index
    Mc = (Np - np.sqrt(np.sum(n_i ** 2))) / Np - np.sqrt(Np)  # McIntosh index
    Hb = (1.0 / Np) * (math.log(np.math.factorial(Np)) - np.sum([math.log(np.math.factorial(int(n_i[i])))
                                             for i in range(n_i.shape[0])]))  # Brillouin index
    Td = np.sum((1.0 / n_i) * (p_i * (1 - p_i)))  # total diversity index
    Ds = (np.sum(n_i * (n_i - 1))) / (Np * (Np - 1))  # The Simpson index
    Bp = np.max(p_i)  # Berger Parker index
    Jidx = H / np.log(S)  # J index
    Ed = Ds / np.log(S)  # Ed index
    HillIdx = (1 / (Ds - 1)) / (np.exp(H) - 1)  # Hill index
    Bg = np.exp(H) / S  # Buzas Gibson index
    CamIdx = 1 - np.sum([(p_i[i] - p_i[i + 1]) / S
                         for i in range(p_i.shape[0] - 1)
                         for j in range(i + 1, p_i.shape[0] - 1)])  # Camargo index

    out_l = [H, Mc, Hb, Td, Ds, Bp,
             Jidx, Ed, HillIdx, Bg, CamIdx]

    out_lbl = ['SD_Shann_Wi_', 'SD_McIntosh_', 'SD_Brillouin_', 'SD_Tot_Diver_', 'SD_Simpson',
               'SD_ BergerParker_', 'SD_J_', 'SD_Ed_', 'SD_Hill_', 'SD_BuzasGibson_', 'SD_Camargo_']

    feat_frame = an_df_filler(feat_frame, row_num, np.ravel(out_l), out_lbl)
    return feat_frame

