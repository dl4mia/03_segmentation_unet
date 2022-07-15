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
    """
        Extracts data from `zip_url` to the location identified by `data_dir` and `project_name` parameters.
        Parameters
        ----------
        zip_url: string
            Indicates the url
        project_name: string
            Indicates the path to the sub-directory at the location identified by the parameter `data_dir`
        data_dir: string
            Indicates the path to the directory where the data should be saved.

    """
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
    """
        Makes directories - `train`, `val, `test` and subdirectories under each `images` and `masks`
        Parameters
        ----------
        data_dir: string
            Indicates the path where the `project` lives.
        project_name: string
            Indicates the name of the sub-folder under the location identified by `data_dir`.
    """
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


def split_train_crops(project_name, center, crops_dir='crops', subset=0.15, by_fraction=True, train_name='train',
                      seed=1000):
    """
    :param project_name: string
           Name of dataset. for example, set this to 'dsb-2018', or 'bbbc010-2012'
    :param center: string
            Set this to 'medoid' or 'centroid'
    :param crops_dir: string
            Name of the crops directory. Default value = 'crops'
    :param subset: int/float
            if by_fraction is True, then subset should be set equal to the percentage of image crops reserved for validation.
            if by_fraction is False, then subset should be set equal to the number of image crops reserved for validation.
    :param by_fraction: boolean
            if True, then reserve a fraction <1 of image crops for validation
    :param train_name: string
            name of directory containing train image and instance crops
    :param seed: int
    :return:
    """
    image_dir = os.path.join(crops_dir, project_name, train_name, 'images')
    instance_dir = os.path.join(crops_dir, project_name, train_name, 'masks')
    center_dir = os.path.join(crops_dir, project_name, train_name, 'center-' + center)

    image_names = sorted(glob(os.path.join(image_dir, '*.tif')))
    instance_names = sorted(glob(os.path.join(instance_dir, '*')))  # this could be `tifs` or `csvs`
    center_names = sorted(glob(os.path.join(center_dir, '*')))  # this could be `tifs` or `csvs`

    indices = np.arange(len(image_names))
    np.random.seed(seed)
    np.random.shuffle(indices)

    if (by_fraction):
        subset_len = int(subset * len(image_names))
    else:
        subset_len = int(subset)

    valIndices = indices[:subset_len]

    image_path_val = os.path.join(crops_dir, project_name, 'val', 'images/')
    instance_path_val = os.path.join(crops_dir, project_name, 'val', 'masks/')
    center_path_val = os.path.join(crops_dir, project_name, 'val', 'center-' + center + '/')

    val_images_exist = False
    val_masks_exist = False
    val_center_images_exist = False

    if not os.path.exists(image_path_val):
        os.makedirs(os.path.dirname(image_path_val))
        print("Created new directory : {}".format(image_path_val))
    else:
        val_images_exist = True

    if not os.path.exists(instance_path_val):
        os.makedirs(os.path.dirname(instance_path_val))
        print("Created new directory : {}".format(instance_path_val))
    else:
        val_masks_exist = True

    if not os.path.exists(center_path_val):
        os.makedirs(os.path.dirname(center_path_val))
        print("Created new directory : {}".format(center_path_val))
    else:
        val_center_images_exist = True

    if not val_images_exist and not val_masks_exist and not val_center_images_exist:
        for val_index in valIndices:
            shutil.move(image_names[val_index], os.path.join(crops_dir, project_name, 'val', 'images'))
            shutil.move(instance_names[val_index], os.path.join(crops_dir, project_name, 'val', 'masks'))
            shutil.move(center_names[val_index], os.path.join(crops_dir, project_name, 'val', 'center-' + center))

        print("Val Images/Masks/Center-{}-image crops saved at {}".format(center,
                                                                          os.path.join(crops_dir, project_name, 'val')))
    else:
        print("Val Images/Masks/Center-{}-image crops already available at {}".format(center, os.path.join(crops_dir,
                                                                                                           project_name,
                                                                                                           'val')))


def split_train_val(data_dir, project_name, train_val_name, subset=0.15, by_fraction=True, seed=1000):
    """
        Splits the `train` directory into `val` directory using the partition percentage of `subset`.
        Parameters
        ----------
        data_dir: string
            Indicates the path where the `project` lives.
        project_name: string
            Indicates the name of the sub-folder under the location identified by `data_dir`.
        train_val_name: string
            Indicates the name of the sub-directory under `project-name` which must be split
        subset: float
            Indicates the fraction of data to be reserved for validation
        seed: integer
            Allows for the same partition to be used in each experiment.
            Change this if you would like to obtain results with different train-val partitions.
    """

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
    """
        Splits the `train` directory into `test` directory using the partition percentage of `subset`.
        Parameters
        ----------
        data_dir: string
            Indicates the path where the `project` lives.
        project_name: string
            Indicates the name of the sub-folder under the location identified by `data_dir`.
        train_test_name: string
            Indicates the name of the sub-directory under `project-name` which must be split
        subset: float
            Indicates the fraction of data to be reserved for evaluation
        seed: integer
            Allows for the same partition to be used in each experiment.
            Change this if you would like to obtain results with different train-test partitions.
    """

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
    """
    :param data_dir: string
            Name of directory storing the data. For example, 'data'
    :param project_name: string
            Name of directroy containing data specific to this project. For example, 'dsb-2018'
    :param train_val_name: string
            Name of directory containing 'train' and 'val' images and instance masks
    :param mode: string
            One of '2d', '3d', '3d_sliced', '3d_ilp'
    :param one_hot: boolean
            set to True, if instances are encoded as one-hot
    :return:
    """
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
        elif (mode in ['3d']):
            ids = np.unique(ma)
            ids = ids[ids != background_id]
            if process_k is not None:
                n_ids = process_k[1]
            else:
                n_ids = len(ids)
            for id in tqdm(ids[:n_ids], position=0, leave=True):
                z, y, x = np.where(ma == id)
                size_list_z.append(np.max(z) - np.min(z))
                size_list_y.append(np.max(y) - np.min(y))
                size_list_x.append(np.max(x) - np.min(x))
                size_list.append(len(x))
        

    print("Minimum object size of the `{}` dataset is equal to {}".format(project_name, np.min(size_list)))
    print("Mean object size of the `{}` dataset is equal to {}".format(project_name, np.mean(size_list)))
    print("Maximum object size of the `{}` dataset is equal to {}".format(project_name, np.percentile(size_list, 99.8)))
    print("Average object size of the `{}` dataset along `x` is equal to {:.3f}".format(project_name,
                                                                                        np.mean(size_list_x)))
    print("Std. dev object size of the `{}` dataset along `x` is equal to {:.3f}".format(project_name,
                                                                                         np.std(size_list_x)))
    print("Average object size of the `{}` dataset along `y` is equal to {:.3f}".format(project_name,
                                                                                        np.mean(size_list_y)))
    print("Std. dev object size of the `{}` dataset along `y` is equal to {:.3f}".format(project_name,
                                                                                         np.std(size_list_y)))

    if mode in ['3d']:
        print("Average object size of the `{}` dataset along `z` is equal to {:.3f}".format(project_name,
                                                                                            np.mean(size_list_z)))
        print("Std. dev object size of the `{}` dataset along `z` is equal to {:.3f}".format(project_name,
                                                                                             np.std(size_list_z)))
        return np.min(size_list).astype(np.float), np.mean(size_list).astype(np.float), np.percentile(size_list, 99.8).astype(np.float), \
               np.mean(size_list_z).astype(np.float), np.mean(size_list_y).astype(
            np.float), np.mean(size_list_x).astype(np.float), \
               np.std(size_list_z).astype(np.float), np.std(size_list_y).astype(np.float), np.std(size_list_x).astype(
            np.float)

    else:
        return np.min(size_list).astype(np.float), np.mean(size_list).astype(np.float), np.percentile(size_list, 99.8).astype(np.float), \
               None, np.mean(size_list_y).astype(np.float), np.mean(size_list_x).astype(np.float), None, np.std(size_list_y).astype(
            np.float), np.std(size_list_x).astype(np.float)


# https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow/59571639#59571639
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_total_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_total_values = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]
    return memory_total_values


def round_up_8(x):
    return (x.astype(int) + 7) & (-8)




def get_data_properties(data_dir, project_name, train_val_name, test_name, mode, process_k=None,
                        anisotropy_factor=1.0, background_id = 0):
    """
    :param data_dir: string
            Path to directory containing all data
    :param project_name: string
            Path to directory containing project-specific images and instances
    :param train_val_name: string
            One of 'train' or 'val'
    :param test_name: string
            Name of test directory.
    :param mode: string
            One of '2d', '3d', '3d_sliced', '3d_ilp'
    :param one_hot: boolean
            set to True, if instances are encoded in a one-hot fashion
    :param process_k (int, int)
            first `int` argument in tuple specifies number of images which must be processed
            second `int` argument in tuple specifies number of ids which must be processed
    :return:
    """
    data_properties_dir = {}
    data_properties_dir['min_object_size'], data_properties_dir['mean_object_size'], data_properties_dir['max_object_size'], \
    data_properties_dir['avg_object_size_z'], data_properties_dir[
        'avg_object_size_y'], data_properties_dir['avg_object_size_x'], \
    data_properties_dir['stdev_object_size_z'], data_properties_dir['stdev_object_size_y'], data_properties_dir[
        'stdev_object_size_x'] = calculate_object_size(data_dir, project_name, train_val_name, mode, process_k, background_id=background_id)
    data_properties_dir['project_name'] = project_name
    return data_properties_dir
