import os
import torch
from scipy import ndimage
from tifffile import imsave
from tqdm import tqdm

from PixelClassification.datasets import get_dataset
from PixelClassification.models import get_model
from PixelClassification.utils2 import matching_dataset

torch.backends.cudnn.benchmark = True
import numpy as np
import torch.nn.functional as F


def test(*args):
    ap_val, min_object_size, model, dataset_it, save_images, save_results, save_dir, = args

    model.eval()

    with torch.no_grad():
        result_list = []
        image_file_names = []
        for idx, sample in tqdm(enumerate(dataset_it)):
            im = sample['image']  # B 1 Y X
            instance = sample['semantic_mask']
            multiple_y = im.shape[2] // 8
            multiple_x = im.shape[3] // 8

            if im.shape[2] % 8 != 0:
                diff_y = 8 * (multiple_y + 1) - im.shape[2]
            else:
                diff_y = 0
            if im.shape[3] % 8 != 0:
                diff_x = 8 * (multiple_x + 1) - im.shape[3]
            else:
                diff_x = 0
            p2d = (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)  # last dim, second last dim

            im = F.pad(im, p2d, "reflect")

            output = model(im)  # B 3 Y X

            output_softmax = F.softmax(output[0], dim=0)
            semantic_pred = torch.argmax(output_softmax, dim=0).detach().cpu().numpy()
            seed_map = output_softmax[1, ...].cpu().detach().numpy()  # Y X
            pred_fg_thresholded = seed_map > 0.5
            instance_map, _ = ndimage.label(pred_fg_thresholded)
            instance_map_filtered = np.zeros_like(instance_map) # Y X

            for item in np.unique(instance_map)[1:]:
                if ((instance_map == item).sum() < min_object_size):
                    instance_map_filtered[instance_map == item] = 0
                else:
                    instance_map_filtered[instance_map == item] = item

            if (diff_y - diff_y // 2) is not 0:
                instance_map_filtered = instance_map_filtered[diff_y // 2:-(diff_y - diff_y // 2), ...]
                seed_map = seed_map[diff_y // 2:-(diff_y - diff_y // 2), ...]
            if (diff_x - diff_x // 2) is not 0:
                instance_map_filtered = instance_map_filtered[..., diff_x // 2:-(diff_x - diff_x // 2)]
                seed_map = seed_map[..., diff_x // 2:-(diff_x - diff_x // 2)]

            result_list.append(np.mean(semantic_pred == instance.detach().cpu().numpy()))
            # results = matching_dataset([instance_map_filtered], [instance[0, 0, ...].cpu().detach().numpy()],
            #                            thresh=ap_val, show_progress=False)
            # print("AP @ {} = {:.3f}".format(str(ap_val), results.accuracy), flush=True)
            # result_list.append(results.accuracy)

            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
            image_file_names.append(base)
            # do for each image

            if save_images and ap_val == 0.5:

                if not os.path.exists(os.path.join(save_dir, 'predictions/')):
                    os.makedirs(os.path.join(save_dir, 'predictions/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'predictions/')))
                
                if not os.path.exists(os.path.join(save_dir, 'semantic_predictions/')):
                    os.makedirs(os.path.join(save_dir, 'semantic_predictions/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'semantic_predictions/')))

                if not os.path.exists(os.path.join(save_dir, 'ground-truth/')):
                    os.makedirs(os.path.join(save_dir, 'ground-truth/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'ground-truth/')))
                if not os.path.exists(os.path.join(save_dir, 'seeds/')):
                    os.makedirs(os.path.join(save_dir, 'seeds/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'seeds/')))

                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                instances_file = os.path.join(save_dir, 'predictions/', base + '.tif')  # TODO
                imsave(instances_file, instance_map_filtered)

                semantic_file = os.path.join(save_dir, 'semantic_predictions/', base + '.tif')  # TODO
                imsave(semantic_file, semantic_pred)

                gt_file = os.path.join(save_dir, 'ground-truth/', base + '.tif')  # TODO
                imsave(gt_file, instance[0, 0, ...].cpu().detach().numpy())
                seed_file = os.path.join(save_dir, 'seeds/', base + '.tif')  # TODO
                imsave(seed_file, seed_map)

            # do for the complete set of images

        print("Mean Result (Accuracy) is {}".format(np.mean(result_list)), flush=True)
    return np.mean(result_list)

def test_3d(*args):
    ap_val, min_object_size, model, dataset_it, save_images, save_results, save_dir, = args

    model.eval()

    with torch.no_grad():
        result_list = []
        image_file_names = []
        for sample in tqdm(dataset_it):

            im = sample['image']  # B 1 Z Y X
            instance= sample['instance_mask']
            multiple_z = im.shape[2] // 8
            multiple_y = im.shape[3] // 8
            multiple_x = im.shape[4] // 8

            if im.shape[2] % 8 != 0:
                diff_z = 8 * (multiple_z + 1) - im.shape[2]
            else:
                diff_z = 0
            if im.shape[3] % 8 != 0:
                diff_y = 8 * (multiple_y + 1) - im.shape[3]
            else:
                diff_y = 0
            if im.shape[4] % 8 != 0:
                diff_x = 8 * (multiple_x + 1) - im.shape[4]
            else:
                diff_x = 0

            p3d = (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2, diff_z // 2, diff_z - diff_z // 2)  # last dim, second last dim

            im = F.pad(im, p3d, "reflect")

            output = model(im) # B 3 Z Y X

            output_softmax = F.softmax(output[0], dim=0)
            seed_map = output_softmax[1, ...].cpu().detach().numpy() # Z Y X
            pred_fg_thresholded = seed_map > 0.5
            instance_map, _ = ndimage.label(pred_fg_thresholded)

            instance_map, nb = ndimage.label(pred_fg_thresholded)
            instance_map_filtered = np.zeros_like(instance_map)

            for item in np.unique(instance_map)[1:]:
                if ((instance_map == item).sum() < min_object_size):
                    instance_map_filtered[instance_map == item] = 0
                else:
                    instance_map_filtered[instance_map == item] = item

            if (diff_z - diff_z // 2) is not 0:
                instance_map_filtered = instance_map_filtered[diff_z // 2:-(diff_z - diff_z // 2), ...]
                seed_map = seed_map[diff_z // 2:-(diff_z - diff_z // 2), ...]
            if (diff_y - diff_y // 2) is not 0:
                instance_map_filtered = instance_map_filtered[:, diff_y // 2:-(diff_y - diff_y // 2), :]
                seed_map = seed_map[:, diff_y // 2:-(diff_y - diff_y // 2), :]
            if (diff_x - diff_x // 2) is not 0:
                instance_map_filtered = instance_map_filtered[..., diff_x // 2:-(diff_x - diff_x // 2)]
                seed_map = seed_map[..., diff_x // 2:-(diff_x - diff_x // 2)]



            results = matching_dataset([instance_map_filtered], [instance[0, 0, ...].cpu().detach().numpy()],
                                       thresh=ap_val, show_progress=False)
            # print("AP @ {} = {:.3f}".format(str(ap_val), results.accuracy), flush=True)
            result_list.append(results.accuracy)
            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
            image_file_names.append(base)
            # do for each image

            if save_images and ap_val == 0.5:

                if not os.path.exists(os.path.join(save_dir, 'predictions/')):
                    os.makedirs(os.path.join(save_dir, 'predictions/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'predictions/')))
                if not os.path.exists(os.path.join(save_dir, 'ground-truth/')):
                    os.makedirs(os.path.join(save_dir, 'ground-truth/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'ground-truth/')))
                if not os.path.exists(os.path.join(save_dir, 'seeds/')):
                    os.makedirs(os.path.join(save_dir, 'seeds/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'seeds/')))

                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                instances_file = os.path.join(save_dir, 'predictions/', base + '.tif')  # TODO
                imsave(instances_file, instance_map_filtered)
                gt_file = os.path.join(save_dir, 'ground-truth/', base + '.tif')  # TODO
                imsave(gt_file, instance[0, 0, ...].cpu().detach().numpy())
                seed_file = os.path.join(save_dir, 'seeds/', base + '.tif')  # TODO
                imsave(seed_file, seed_map)

            # do for the complete set of images

        print("Mean Result (Accuracy) is {}".format(np.mean(result_list)), flush=True)
    return np.mean(result_list)

def begin_evaluating(test_configs):
    ap_val = test_configs['ap_val']
    min_object_size = test_configs['min_object_size']
    save_images = test_configs['save_images']
    save_results = test_configs['save_results']
    save_dir = test_configs['save_dir']

    # set device
    device = torch.device("cuda:0" if test_configs['cuda'] else "cpu")

    # dataloader
    dataset = get_dataset(test_configs['dataset']['name'], test_configs['dataset']['kwargs'])
    dataset_it = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4,
                                             pin_memory=True if test_configs['cuda'] else False)

    # load model
    model = get_model(test_configs['model']['name'], test_configs['model']['kwargs'])
    model = torch.nn.DataParallel(model).to(device)

    # load snapshot
    if os.path.exists(test_configs['checkpoint_path']):
        state = torch.load(test_configs['checkpoint_path'])
        model.load_state_dict(state['model_state_dict'], strict=True)
    else:
        assert False, 'checkpoint_path {} does not exist!'.format(test_configs['checkpoint_path'])

    # test on evaluation images:

    if (test_configs['name'] == '2d'):
        args = (ap_val, min_object_size, model, dataset_it, save_images, save_results, save_dir)

        result = test(*args)

    elif (test_configs['name'] == '3d'):
        args = (ap_val, min_object_size, model, dataset_it, save_images, save_results, save_dir)

        result = test_3d(*args)

    return result
