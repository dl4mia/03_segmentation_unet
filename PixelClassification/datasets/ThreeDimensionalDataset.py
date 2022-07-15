import glob
import numpy as np
import os
import random
import tifffile
from skimage.segmentation import find_boundaries
from skimage.segmentation import relabel_sequential
from torch.utils.data import Dataset


class ThreeDimensionalDataset(Dataset):
    def __init__(self, data_dir, type, bg_id=0, size=None, crop_z=32, crop_y=64, crop_x=64, transform=None):
        print('3D `{}` Dataset created'.format(type))
        # get image and instance list

        image_list = sorted(glob.glob(os.path.join(data_dir, '{}/'.format(type), 'images/*.tif')))
        self.image_list = image_list

        instance_list = sorted(glob.glob(os.path.join(data_dir, '{}/'.format(type), 'masks/*.tif')))
        self.instance_list = instance_list

        self.bg_id = bg_id
        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform
        self.type = type
        self.crop_z = crop_z
        self.crop_y = crop_y
        self.crop_x = crop_x
        print("Number of 3D images discovered is {}".format(self.real_size))

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}
        image = tifffile.imread(self.image_list[index])  # Z Y X
        mask = tifffile.imread(self.instance_list[index])  # Z Y X
        image_normalized = self.normalize(image[np.newaxis, ...], axis=(1, 2, 3))  # added new axis already for channel
        class_map, instance_map = self.convert_instance_to_class_ids(mask)

        if self.type == 'train' or self.type == 'val':
            inside = 0
            while not inside:
                z_start = np.random.randint(image.shape[0])
                y_start = np.random.randint(image.shape[1])
                x_start = np.random.randint(image.shape[2])
                if z_start + self.crop_z <= image.shape[0] and y_start + self.crop_y <= image.shape[
                    1] and x_start + self.crop_x <= image.shape[2] and len(
                        np.unique(instance_map[z_start:z_start + self.crop_z,
                                  y_start:y_start + self.crop_y, x_start:x_start + self.crop_x])) > 1:
                    sample['image'] = image_normalized[:, z_start:z_start + self.crop_z, y_start:y_start + self.crop_y,
                                      x_start:x_start + self.crop_x]
                    sample['semantic_mask'] = class_map[np.newaxis, z_start:z_start + self.crop_z,
                                              y_start:y_start + self.crop_y,
                                              x_start:x_start + self.crop_x]  # 1 Z Y X
                    sample['instance_mask'] = instance_map[np.newaxis, z_start:z_start + self.crop_z,
                                              y_start:y_start + self.crop_y, x_start:x_start + self.crop_x]  # 1 Z Y X
                    inside = 1
        else:
            sample['image'] = image_normalized
            sample['semantic_mask'] = class_map[np.newaxis, ...]
            sample['instance_mask'] = instance_map[np.newaxis, ...]

        sample['im_name'] = self.image_list[index]
        if (self.transform is not None):
            return self.transform(sample)
        else:
            return sample

    def __len__(self):
        return self.real_size if self.size is None else self.size

    @classmethod
    def convert_instance_to_class_ids(cls, pic, bg_id=0):
        class_map = convert_to_class_labels(pic)
        instance_map = np.zeros(pic.shape, dtype=np.int16)
        mask_fg = pic > bg_id
        if mask_fg.sum() > 0:
            ids, _, _ = relabel_sequential(pic[mask_fg])
            instance_map[mask_fg] = ids
        return class_map, instance_map

    @classmethod
    def normalize(cls, pic, pmin=1, pmax=99.8, axis=(1, 2), clip=False, eps=1e-20, dtype=np.float32):
        mi = np.percentile(pic, pmin, axis=axis, keepdims=True)
        ma = np.percentile(pic, pmax, axis=axis, keepdims=True)
        return cls.normalize_mi_ma(pic, mi, ma, clip=clip, eps=eps, dtype=dtype)

    @classmethod
    def normalize_mi_ma(cls, x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
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


def convert_to_class_labels(lbl):
    b = find_boundaries(lbl, mode='outer')
    res = (lbl > 0).astype(np.uint8)
    res[b] = 2
    return res
