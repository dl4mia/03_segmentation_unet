import os
import torch

import PixelClassification.utils.transforms as my_transforms


def create_dataset_dict(data_dir,
                        project_name,
                        size,
                        type,
                        name='2d',
                        batch_size=16,
                        workers=4,

                        ):
    if name == '2d':
        set_transforms = my_transforms.get_transform([
            {
                'name': 'RandomRotationsAndFlips',
                'opts': {
                    'keys': ('image', 'instance_mask', 'semantic_mask'),
                    'degrees': 90,

                }
            },
            {
                'name': 'ToTensorFromNumpy',
                'opts': {
                    'keys': ('image', 'instance_mask', 'semantic_mask'),
                    'type': (torch.FloatTensor, torch.ShortTensor, torch.ShortTensor),
                }
            },
        ])
    elif name == '3d':
        set_transforms = my_transforms.get_transform([
            {
                'name': 'RandomRotationsAndFlips3D',
                'opts': {
                    'keys': ('image', 'instance_mask', 'semantic_mask'),
                    'degrees': 90,

                }
            },
            {
                'name': 'ToTensorFromNumpy',
                'opts': {
                    'keys': ('image', 'instance_mask', 'semantic_mask'),
                    'type': (torch.FloatTensor, torch.ShortTensor, torch.ShortTensor),
                }
            },
        ])
    dataset_dict = {
        'name': name,
        'kwargs': {
            'data_dir': os.path.join(data_dir, project_name),
            'type': type,
            'size': size,
            'transform': set_transforms,
            
        },
        'batch_size': batch_size,
        'workers': workers,

    }
    print("`{}_dataset_dict` dictionary successfully created with: \n -- {} images accessed from {}, "
          "\n -- number of images per epoch equal to {}, "
          "\n -- batch size set at {}, "
          .format(type, type, os.path.join(data_dir, project_name, type, 'images'), size, batch_size))
    return dataset_dict


def create_test_configs_dict(data_dir,
                             checkpoint_path,
                             save_dir=None,
                             ap_val=0.5,
                             min_object_size=10,
                             save_images=True,
                             save_results=True,
                             cuda=True,
                             name='2d',
                             num_classes=3,
                             type='test',
                             ):
    if name == '2d':
        model_name = 'unet'
    elif name == '3d':
        model_name = 'unet3d'

    test_configs = dict(
        ap_val=ap_val,
        min_object_size=min_object_size,
        cuda=cuda,
        save_results=save_results,
        save_images=save_images,
        save_dir=save_dir,
        checkpoint_path=checkpoint_path,
        name=name,
        dataset={
            'name': name,
            'kwargs': {
                'data_dir': data_dir,
                'type': type,

                'transform': my_transforms.get_transform([
                    {
                        'name': 'ToTensorFromNumpy',
                        'opts': {
                            'keys': ('image', 'instance_mask', 'semantic_mask'),
                            'type': (torch.FloatTensor, torch.ShortTensor, torch.ShortTensor),
                        }
                    },
                ]),
            }
        },

        model={
            'name': model_name,
            'kwargs': {
                'num_classes': num_classes,
            }
        }
    )
    print(
        "`test_configs` dictionary successfully created with: "
        "\n -- evaluation images accessed from {}, "
        "\n -- trained weights accessed from {}, "
        "\n -- output directory chosen as {}".format(
            data_dir, checkpoint_path, save_dir))
    return test_configs


def create_model_dict(num_classes=3, depth=3, in_channels=1, name='2d'):
    model_dict = {
        'name': 'unet' if name == '2d' else 'unet3d',
        'kwargs': {
            'num_classes': num_classes,
            'depth': depth,
            'in_channels': in_channels,

        }
    }
    print(
        "`model_dict` dictionary successfully created with: \n -- num of classes equal to {}, \n -- name equal to {}".format(
            num_classes, model_dict['name']))
    return model_dict


def create_configs(save_dir,
                   resume_path,
                   n_epochs=200,
                   train_lr=5e-4,
                   cuda=True,
                   save=True,
                   ):
    configs = dict(train_lr=train_lr,
                   n_epochs=n_epochs,
                   cuda=cuda,
                   save=save,
                   save_dir=save_dir,
                   resume_path=resume_path,
                   )
    print(
        "`configs` dictionary successfully created with: "
        "\n -- n_epochs equal to {}, "
        "\n -- save_dir equal to {}, "
        .format(n_epochs, save_dir))
    return configs
