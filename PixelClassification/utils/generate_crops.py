import numpy as np
import os
import tifffile
from numba import jit
from scipy.ndimage import zoom
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import binary_fill_holes
import pandas as pd
from tqdm import tqdm


def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img == l
        mask_filled = binary_fill_holes(mask, **kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def fill_label_holes(lbl_img, **kwargs):
    """
        Fill small holes in label image.
    """

    def grow(sl, interior):
        return tuple(slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior))

    def shrink(interior):
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i, sl in enumerate(objects, 1):
        if sl is None: continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = binary_fill_holes(grown_mask, **kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def normalize_min_max_percentile(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """
        Percentile-based image normalization.
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


def process(im, inst, crops_dir, data_subset, crop_size,  norm='min-max-percentile', fraction_max_ids = 1.0, background_id=0):
    image_path = os.path.join(crops_dir, data_subset, 'images/')
    instance_path = os.path.join(crops_dir, data_subset, 'masks/')

    if not os.path.exists(image_path):
        os.makedirs(os.path.dirname(image_path))
        print("Created new directory : {}".format(image_path))
    if not os.path.exists(instance_path):
        os.makedirs(os.path.dirname(instance_path))
        print("Created new directory : {}".format(instance_path))

    instance = tifffile.imread(inst).astype(np.uint16)
    image = tifffile.imread(im).astype(np.float32)

    if image.ndim == 2:  # gray-scale
        image = normalize_min_max_percentile(image, 1, 99.8, axis=(0, 1))

    instance = fill_label_holes(instance)

    if image.ndim == 2:
        h, w = image.shape
    instance_np = np.array(instance, copy=False)

    object_mask = instance_np > background_id
    # ensure that background is mapped to 0
    instance_np[instance_np == background_id] = 0
    ids = np.unique(instance_np[object_mask])
    ids = ids[ids != 0]
    ids_subset = np.random.choice(ids, int(fraction_max_ids * len(ids)), replace=False)

    # loop over instances
    for j, id in enumerate(ids_subset):
        y, x = np.where(instance_np == id)
        ym, xm = np.mean(y), np.mean(x)

        jj = int(np.clip(ym - crop_size / 2, 0, h - crop_size))
        ii = int(np.clip(xm - crop_size / 2, 0, w - crop_size))
        if image.ndim == 2:
            if (image[jj:jj + crop_size, ii:ii + crop_size].shape == (crop_size, crop_size)):
                im_crop = image[jj:jj + crop_size, ii:ii + crop_size]
                instance_crop = instance_np[jj:jj + crop_size, ii:ii + crop_size]
                tifffile.imsave(image_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j), im_crop)
                tifffile.imsave(instance_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j),
                                instance_crop)
                
        
def process_3d(im, inst, crops_dir, data_subset, crop_size_x, crop_size_y, crop_size_z, norm='min-max-percentile',
               fraction_max_ids = 1.0, anisotropy_factor=1.0, speed_up=2.0, background_id = 0):
    


    image_path = os.path.join(crops_dir, data_subset, 'images/')
    instance_path = os.path.join(crops_dir, data_subset, 'masks/')
    
    if not os.path.exists(image_path):
        os.makedirs(os.path.dirname(image_path))
        print("Created new directory : {}".format(image_path))
    if not os.path.exists(instance_path):
        os.makedirs(os.path.dirname(instance_path))
        print("Created new directory : {}".format(instance_path))
    
    instance = tifffile.imread(inst).astype(np.uint16)
    image = tifffile.imread(im).astype(np.float32)

    image = normalize_min_max_percentile(image, 1, 99.8, axis=(0, 1, 2))
    instance = fill_label_holes(instance)

    d, h, w = image.shape
    instance_np = np.array(instance, copy=False)
    object_mask = instance_np > background_id
    # ensure that background is mapped to 0
    instance_np[instance_np==background_id] = 0
    ids = np.unique(instance_np[object_mask])
    ids = ids[ids != 0]
    ids_subset = np.random.choice(ids, int(fraction_max_ids * len(ids)), replace=False)
    # loop over instances
    for j, id in enumerate(ids_subset):
        z, y, x = np.where(instance_np == id)
        zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
        kk = int(np.clip(zm - crop_size_z / 2, 0, d - crop_size_z))
        jj = int(np.clip(ym - crop_size_y / 2, 0, h - crop_size_y))
        ii = int(np.clip(xm - crop_size_x / 2, 0, w - crop_size_x))

        if (image[kk:kk + crop_size_z, jj:jj + crop_size_y, ii:ii + crop_size_x].shape == (
                crop_size_z, crop_size_y, crop_size_x)):
            im_crop = image[kk:kk + crop_size_z, jj:jj + crop_size_y, ii:ii + crop_size_x]
            instance_crop = instance_np[kk:kk + crop_size_z, jj:jj + crop_size_y, ii:ii + crop_size_x]
    
            tifffile.imsave(image_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j), im_crop)
            tifffile.imsave(instance_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j),
                                instance_crop.astype(np.uint16))
            

