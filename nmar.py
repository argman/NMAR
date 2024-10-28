
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pydicom as dicom

from leapctype import *
leapct = tomographicModels()
leapct.about()

from sklearn.cluster import KMeans
from skimage.filters import gaussian


# the linear attenuation coefficient of water [1/cm], this may change for different scans, please adjust it
MIU_AIR = 0
MIU_WATER = 0.19


class LeapWrapper:
    def __init__(self, img_size):
        # Specify the number of detector columns which is used below
        # Scale the number of angles and the detector pixel size with N
        numCols = img_size
        numAngles = 2*2*int(360*numCols/1024)
        pixelSize = 1

        # Set the number of detector rows
        numRows = 1
        
        leapct.set_fanbeam(numAngles=numAngles, numRows=numRows, numCols=numCols,
            pixelHeight=pixelSize, pixelWidth=pixelSize,
            centerRow=0.5*(numRows-1), centerCol=0.5*(numCols-1),
            phis=leapct.setAngleArray(numAngles, 360.0),
            sod=1075, sdd=1075 + img_size//2 + 1)
        leapct.set_default_volume()
        self.g = leapct.allocate_projections() # shape is numAngles, numRows, numCols
        self.f = leapct.allocate_volume() # shape is numZ, numY, numX

    def fanbeam(self, x):
        self.f[0] = x
        startTime = time.time()
        leapct.project(self.g, self.f)
        # leapct.rayTrace(self.g)
        print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))
        return self.g[:, 0].copy()

    def ifanbeam(self, x):
        self.f[:] = 0
        self.g[:, 0] = x
        leapct.FBP(self.g, self.f)
        return self.f[0]


def proj_interp(proj, metal_trace):
    num_of_view, num_of_bin = proj.shape
    p_interp = np.zeros_like(proj)

    for i in range(num_of_view):
        mslice = metal_trace[i]
        pslice = proj[i].copy()

        metal_pos = np.nonzero(mslice)[0]
        non_metal_pos = np.where(mslice == 0)[0]

        pslice[metal_pos] = np.interp(metal_pos, non_metal_pos, pslice[non_metal_pos])

        p_interp[i] = pslice

    return p_interp


def nmar_proj_interp(proj, proj_prior, metal_trace):
    '''
    see "Normalized metal artifact reduction (NMAR) in computed tomography"
    link https://pubmed.ncbi.nlm.nih.gov/21089784/
    '''
    proj_prior[proj_prior < 0] = 0
    eps = 1e-6
    proj_prior = proj_prior + eps
    proj_norm = proj / proj_prior
    proj_norm_interp = proj_interp(proj_norm, metal_trace)
    proj_nmar = proj_norm_interp * proj_prior
    proj_nmar[metal_trace == 0] = proj[metal_trace == 0]

    return proj_nmar


def linear_attenuation(im, reverse=False):
    if not reverse:
        return 0.19 * (1 + im / 1000)
    else:
        return (im / 0.19 - 1) * 1000


def circle_mask(image):
    shape_min = min(image.shape)
    radius = shape_min // 2
    img_shape = np.array(image.shape)
    coords = np.array(np.ogrid[:image.shape[0], :image.shape[1]],
                        dtype=object)
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    outside_reconstruction_circle = dist > radius ** 2
    return outside_reconstruction_circle


def get_metal(im):
    metal = im > 2500
    return metal


def mar(im, show_result=True):
    '''
    param:
        im: two dimensinal numpy array
    '''
    H, W = im.shape
    if H != W:
        print('H and W not the same')
        return im
    
    metal_bw = get_metal(im)
    if np.sum(metal_bw) < 10:
        print('Skip as no metal found')
        return im
    cm = circle_mask(im)

    leap_trans = LeapWrapper(max(H, W))

    im_raw = linear_attenuation(im.copy())
    im_raw[cm] = MIU_AIR
    
    # uncorrected projection
    proj = leap_trans.fanbeam(im_raw)
    # metal image(binary image)
    proj_metal = leap_trans.fanbeam(metal_bw)
    # metal trace in projection domain(binary image)
    metal_trace = proj_metal > 0

    # perform linear interpolation correction
    proj_li_corr = proj_interp(proj, metal_trace)
    im_li = leap_trans.ifanbeam(proj_li_corr)

    # nmar basic
    # im_raw[metal_bw > 0] = MIU_WATER
    # im_raw[im_raw < MIU_AIR] = MIU_AIR
    # model = KMeans(n_clusters=3, init=[[MIU_AIR], [MIU_WATER], [2 * MIU_WATER]]).fit(im_raw.reshape(-1, 1))
    # src_label = model.predict(im_raw.reshape(-1, 1)).reshape(im_raw.shape)
    # thresh_bone = max(1.2 * MIU_WATER, np.min(im_raw[src_label == 2]))
    # print('Threshold bone {}'.format(thresh_bone))
    # thresh_water = np.min(im_raw[src_label == 1])
    # print('Threshold water {}'.format(thresh_water))
    # im_raw_smooth = gaussian(im_raw, 1)
    # prior_img = im_raw_smooth.copy()
    # prior_img[im_raw_smooth <= thresh_water] = MIU_AIR
    # prior_img[(im_raw_smooth > thresh_water) & (im_raw_smooth < thresh_bone)] = MIU_WATER
    # proj_prior = radon(prior_img, thetas, **radon_params)
    # proj_nmar1 = nmar_proj_interp(proj, proj_prior, metal_trace)
    # im_nmar1 = iradon(proj_nmar1, thetas, **iradon_params)

    # nmar 2 based on li
    im_li[metal_bw > 0] = MIU_WATER
    model = KMeans(n_clusters=3, init=[[MIU_AIR], [MIU_WATER], [2 * MIU_WATER]]).fit(im_li.reshape(-1, 1))
    src_label = model.predict(im_li.reshape(-1, 1)).reshape(im_li.shape)
    thresh_bone = max(1.2 * MIU_WATER, np.min(im_li[src_label == 2]))
    print('Threshold bone {}'.format(thresh_bone))
    thresh_water = np.min(im_li[src_label == 1])
    print('Threshold water {}'.format(thresh_water))
    im_li_smooth = gaussian(im_li, 1)
    prior_img = im_li_smooth.copy()
    prior_img[im_li_smooth <= thresh_water] = MIU_AIR
    prior_img[(im_li_smooth > thresh_water) & (im_li_smooth < thresh_bone)] = MIU_WATER
    # proj_prior = radon(prior_img, thetas, **radon_params)
    proj_prior = leap_trans.fanbeam(prior_img)
    proj_nmar2 = nmar_proj_interp(proj, proj_prior, metal_trace)
    im_nmar2 = leap_trans.ifanbeam(proj_nmar2)

    result = linear_attenuation(im_nmar2, True)
    result[metal_bw] = im[metal_bw]
    result[cm] = im[cm]
    
    if show_result:
        fig, axs = plt.subplots(1, 4)
        axs[0].set_title('Original')
        axs[0].imshow(im, vmin=-80, vmax=160)
        axs[1].set_title('Linear Interpolation Correction')
        axs[1].imshow(np.clip(linear_attenuation(im_li, True).astype(np.int16), -80, 160))
        axs[2].set_title('NMAR 2')
        axs[2].imshow(np.clip(result.astype(np.int16), -80, 160))
        axs[3].set_title('Difference')
        axs[3].imshow(np.abs(result - im))
        plt.show()

    return result


if __name__ == '__main__':
    im = np.load(r'E:\PycharmProjects\mar\NMAR\data/lung.npy')

    mar(im, show_result=True)