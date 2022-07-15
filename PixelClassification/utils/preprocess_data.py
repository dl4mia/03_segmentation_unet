import numpy as np
import os
import shutil
import subprocess as sp
import tifffile
import urllib.request
import zipfile
from glob import glob
from tqdm import tqdm


def extract_data(zip_url, project_name, data_dir='../../../data/'):
    zip_path = os.path.join(data_dir, project_name + '.zip')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created new directory {}".format(data_dir))

    if (os.path.exists(zip_path)):
        print("Zip file was downloaded and extracted before!")
    else:
        if (os.path.exists(os.path.join(data_dir, project_name, 'download/'))):
            pass
        else:
            os.makedirs(os.path.join(data_dir, project_name, 'download/'))
            urllib.request.urlretrieve(zip_url, zip_path)
            print("Downloaded data as {}".format(zip_path))
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            if os.path.exists(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'train')):
                shutil.move(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'train'),
                            os.path.join(data_dir, project_name, 'download/'))
            if os.path.exists(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'val')):
                shutil.move(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'val'),
                            os.path.join(data_dir, project_name, 'download/'))
            if os.path.exists(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'test')):
                shutil.move(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'test'),
                            os.path.join(data_dir, project_name, 'download/'))
            print("Unzipped data to {}".format(os.path.join(data_dir, project_name, 'download/')))


def make_dirs(data_dir, project_name):
    image_path_train = os.path.join(data_dir, project_name, 'train', 'images/')
    instance_path_train = os.path.join(data_dir, project_name, 'train', 'masks/')
    image_path_val = os.path.join(data_dir, project_name, 'val', 'images/')
    instance_path_val = os.path.join(data_dir, project_name, 'val', 'masks/')
    image_path_test = os.path.join(data_dir, project_name, 'test', 'images/')
    instance_path_test = os.path.join(data_dir, project_name, 'test', 'masks/')

    if not os.path.exists(image_path_train):
        os.makedirs(os.path.dirname(image_path_train))
        print("Created new directory : {}".format(image_path_train))

    if not os.path.exists(instance_path_train):
        os.makedirs(os.path.dirname(instance_path_train))
        print("Created new directory : {}".format(instance_path_train))

    if not os.path.exists(image_path_val):
        os.makedirs(os.path.dirname(image_path_val))
        print("Created new directory : {}".format(image_path_val))

    if not os.path.exists(instance_path_val):
        os.makedirs(os.path.dirname(instance_path_val))
        print("Created new directory : {}".format(instance_path_val))

    if not os.path.exists(image_path_test):
        os.makedirs(os.path.dirname(image_path_test))
        print("Created new directory : {}".format(image_path_test))

    if not os.path.exists(instance_path_test):
        os.makedirs(os.path.dirname(instance_path_test))
        print("Created new directory : {}".format(instance_path_test))


def split_train_val(data_dir, project_name, train_val_name, subset=0.15, by_fraction=True, seed=1000):
    image_dir = os.path.join(data_dir, project_name, 'download', train_val_name, 'images')
    instance_dir = os.path.join(data_dir, project_name, 'download', train_val_name, 'masks')
    image_names = sorted(glob(os.path.join(image_dir, '*.tif')))
    instance_names = sorted(glob(os.path.join(instance_dir, '*.tif')))
    indices = np.arange(len(image_names))
    np.random.seed(seed)
    np.random.shuffle(indices)
    if (by_fraction):
        subset_len = int(subset * len(image_names))
    else:
        subset_len = int(subset)
    val_indices = indices[:subset_len]
    trainIndices = indices[subset_len:]
    make_dirs(data_dir=data_dir, project_name=project_name)

    for val_index in val_indices:
        shutil.copy(image_names[val_index], os.path.join(data_dir, project_name, 'val', 'images'))
        shutil.copy(instance_names[val_index], os.path.join(data_dir, project_name, 'val', 'masks'))

    for trainIndex in trainIndices:
        shutil.copy(image_names[trainIndex], os.path.join(data_dir, project_name, 'train', 'images'))
        shutil.copy(instance_names[trainIndex], os.path.join(data_dir, project_name, 'train', 'masks'))

    image_dir = os.path.join(data_dir, project_name, 'download', 'test', 'images')
    instance_dir = os.path.join(data_dir, project_name, 'download', 'test', 'masks')
    image_names = sorted(glob(os.path.join(image_dir, '*.tif')))
    instance_names = sorted(glob(os.path.join(instance_dir, '*.tif')))
    test_indices = np.arange(len(image_names))
    for test_index in test_indices:
        shutil.copy(image_names[test_index], os.path.join(data_dir, project_name, 'test', 'images'))
        shutil.copy(instance_names[test_index], os.path.join(data_dir, project_name, 'test', 'masks'))
    print("Train-Val-Test Images/Masks copied to {}".format(os.path.join(data_dir, project_name)))


def split_train_test(data_dir, project_name, train_test_name, subset=0.5, by_fraction=True, seed=1000):
    image_dir = os.path.join(data_dir, project_name, 'download', train_test_name, 'images')
    instance_dir = os.path.join(data_dir, project_name, 'download', train_test_name, 'masks')
    image_names = sorted(glob(os.path.join(image_dir, '*.tif')))
    instance_names = sorted(glob(os.path.join(instance_dir, '*.tif')))
    indices = np.arange(len(image_names))
    np.random.seed(seed)
    np.random.shuffle(indices)
    if (by_fraction):
        subset_len = int(subset * len(image_names))
    else:
        subset_len = int(subset)
    test_indices = indices[:subset_len]
    # make_dirs(data_dir=data_dir, project_name=project_name)
    test_images_exist = False
    test_masks_exist = False
    if not os.path.exists(os.path.join(data_dir, project_name, 'download', 'test', 'images')):
        os.makedirs(os.path.join(data_dir, project_name, 'download', 'test', 'images'))
        print("Created new directory : {}".format(os.path.join(data_dir, project_name, 'download', 'test', 'images')))
    else:
        test_images_exist = True
    if not os.path.exists(os.path.join(data_dir, project_name, 'download', 'test', 'masks')):
        os.makedirs(os.path.join(data_dir, project_name, 'download', 'test', 'masks'))
        print("Created new directory : {}".format(os.path.join(data_dir, project_name, 'download', 'test', 'masks')))
    else:
        test_masks_exist = True
    if not test_images_exist and not test_masks_exist:
        for test_index in test_indices:
            shutil.move(image_names[test_index], os.path.join(data_dir, project_name, 'download', 'test', 'images'))
            shutil.move(instance_names[test_index], os.path.join(data_dir, project_name, 'download', 'test', 'masks'))
        print("Train-Test Images/Masks saved at {}".format(os.path.join(data_dir, project_name, 'download')))
    else:
        print(
            "Train-Test Images/Masks already available at {}".format(os.path.join(data_dir, project_name, 'download')))


def calculate_object_size(data_dir, project_name, train_val_name, mode, process_k, background_id=0):
    instance_names = []
    size_list_x = []
    size_list_y = []
    size_list_z = []
    size_list = []
    for name in train_val_name:
        instance_dir = os.path.join(data_dir, project_name, name, 'masks')
        instance_names += sorted(glob(os.path.join(instance_dir, '*.tif')))

    if process_k is not None:
        n_images = process_k[0]
    else:
        n_images = len((instance_names))
    for i in tqdm(range(len(instance_names[:n_images])), position=0, leave=True):
        ma = tifffile.imread(instance_names[i])
        if (mode == '2d'):
            ids = np.unique(ma)
            ids = ids[ids != background_id]
            for id in ids:
                y, x = np.where(ma == id)
                size_list_x.append(np.max(x) - np.min(x))
                size_list_y.append(np.max(y) - np.min(y))
                size_list.append(len(x))
        elif (mode in '3d'):
            ids = np.unique(ma)
            ids = ids[ids != background_id]
            if process_k is not None:
                n_ids = process_k[1]
            else:
                n_ids = len(ids)
            for id in tqdm(ids[:n_ids], position=0, leave=True):
                # for id in ids:
                z, y, x = np.where(ma == id)
                size_list_z.append(np.max(z) - np.min(z))
                size_list_y.append(np.max(y) - np.min(y))
                size_list_x.append(np.max(x) - np.min(x))
                size_list.append(len(x))

    print("Minimum object size of the `{}` dataset is equal to {}.".format(project_name, np.min(size_list)))
    return np.min(size_list).astype(np.float)


def get_data_properties(data_dir, project_name, train_val_name, mode, process_k=None, background_id=0):
    data_properties_dir = {}
    data_properties_dir['min_object_size'] = calculate_object_size(data_dir, project_name, train_val_name, mode,
                                                                   process_k, background_id=background_id)
    data_properties_dir['project_name'] = project_name
    return data_properties_dir
