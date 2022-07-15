import ast
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tifffile
from glob import glob
from scipy.ndimage import zoom
from skimage.segmentation import relabel_sequential
from PixelClassification.utils.glasbey import Glasbey
from skimage.segmentation import find_boundaries

def create_color_map(n_colors=10):
    gb = Glasbey(base_palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
                 lightness_range=(10, 100),
                 hue_range=(10, 100),
                 chroma_range=(10, 100),
                 no_black=True)
    p = gb.generate_palette(size=n_colors)
    p[0, :] = [0, 0, 0]  # make label 0 always black!
    p_ = np.hstack((p, np.ones((p.shape[0], 1))))
    p_ = np.where(p_ > 0, p_, 0)
    p_ = np.where(p_ <= 1, p_, 1)
    return p_


def visualize(image, prediction, ground_truth, seed, new_cmp):
    font = {'family': 'serif',
            'color': 'white',
            'weight': 'bold',
            'size': 16,
            }
    plt.figure(figsize=(15, 15))
    img_show = image if image.ndim == 2 else image[0, ...]
    plt.subplot(221);
    plt.imshow(img_show, cmap='magma');
    plt.text(30, 30, "IM", fontdict=font)
    plt.xlabel('Image')
    plt.axis('off')
    if (ground_truth is not None):
        plt.subplot(222);
        plt.axis('off')
        plt.imshow(ground_truth, cmap=new_cmp, interpolation='None')
        plt.text(30, 30, "GT", fontdict=font)
        plt.xlabel('Ground Truth')
    plt.subplot(223);
    plt.axis('off')
    plt.imshow(seed, interpolation='None')
    plt.subplot(224);
    plt.axis('off')
    plt.imshow(prediction, cmap=new_cmp, interpolation='None')
    plt.text(30, 30, "PRED", fontdict=font)
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()


def decode(filename, one_hot=False, center=False):
    df = pd.read_csv(filename, header=None)
    df_numpy = df.to_numpy()
    d = {}

    if one_hot:
        mask_decoded = []
        for row in df_numpy:
            d['counts'] = ast.literal_eval(row[1])
            d['size'] = ast.literal_eval(row[2])
            mask = rletools.decode(d)  # returns binary mask
            mask_decoded.append(mask)
    else:
        if center:
            mask_decoded = np.zeros(ast.literal_eval(df_numpy[0][2]),
                                    dtype=np.bool)  # obtain size by reading first row of csv file
        else:
            mask_decoded = np.zeros(ast.literal_eval(df_numpy[0][2]),
                                    dtype=np.uint16)  # obtain size by reading first row of csv file
        for row in df_numpy:
            d['counts'] = ast.literal_eval(row[1])
            d['size'] = ast.literal_eval(row[2])
            mask = rletools.decode(d)  # returns binary mask
            y, x = np.where(mask == 1)
            mask_decoded[y, x] = int(row[0])
    return np.asarray(mask_decoded)




def visualize_3d(im_filename, gt_filename, pred_filename, seed_filename, new_cmp, anisotropy):
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'bold',
            'size': 16,
            }
    fig = plt.figure(constrained_layout=False, figsize=(12, 10))
    # rows represent three views of the sstack. cols represent the image, center and label
    spec = gridspec.GridSpec(ncols=3, nrows=4, figure=fig)
    im = tifffile.imread(im_filename)
    gt_label, _, _ = relabel_sequential(tifffile.imread(gt_filename))
    pred_label, _, _ = relabel_sequential(tifffile.imread(pred_filename))
    seed = tifffile.imread(seed_filename)

    im = zoom(im, (anisotropy, 1, 1), order=0)
    gt_label = zoom(gt_label, (anisotropy, 1, 1), order=0)
    pred_label = zoom(pred_label, (anisotropy, 1, 1), order=0)
    seed = zoom(seed, (anisotropy, 1, 1), order=0)

    z_mid = im.shape[0] // 2
    y_mid = im.shape[1] // 2
    x_mid = im.shape[2] // 2

    ax0 = fig.add_subplot(spec[0, 0])
    ax0.imshow(im[z_mid, ...], cmap='magma', interpolation='None')
    ax0.set_yticklabels([])
    ax0.set_yticks([])
    ax0.set_xticklabels([])
    ax0.set_xticks([])
    ax0.set_ylabel('IM', fontdict=font)
    ax0.set_xlabel('xy', fontdict=font)
    ax0.xaxis.set_label_position('top')

    ax1 = fig.add_subplot(spec[1, 0])
    ax1.imshow(gt_label[z_mid, ...], cmap=new_cmp, interpolation='None')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax1.set_ylabel('GT', fontdict=font)

    ax2 = fig.add_subplot(spec[2, 0])
    ax2.imshow(pred_label[z_mid, ...], cmap=new_cmp, interpolation='None')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    ax2.set_ylabel('PRED', fontdict=font)

    ax3 = fig.add_subplot(spec[3, 0])
    ax3.imshow(seed[z_mid, ...], cmap='magma', interpolation='None')
    ax3.axes.get_xaxis().set_visible(False)
    ax3.set_yticklabels([])
    ax3.set_yticks([])
    ax3.set_ylabel('SEED', fontdict=font)



    # use interpolation
    ax8 = fig.add_subplot(spec[0, 1])
    ax8.imshow(im[..., x_mid], cmap='magma', interpolation='None')
    ax8.set_yticklabels([])
    ax8.set_yticks([])
    ax8.set_xticklabels([])
    ax8.set_xticks([])
    ax8.set_xlabel('yz', fontdict=font)
    ax8.xaxis.set_label_position('top')

    ax9 = fig.add_subplot(spec[1, 1])
    ax9.imshow(gt_label[..., x_mid], cmap=new_cmp, interpolation='None')
    ax9.axes.get_xaxis().set_visible(False)
    ax9.set_yticklabels([])
    ax9.set_yticks([])

    ax10 = fig.add_subplot(spec[2, 1])
    ax10.imshow(pred_label[..., x_mid], cmap=new_cmp, interpolation='None')
    ax10.axes.get_xaxis().set_visible(False)
    ax10.set_yticklabels([])
    ax10.set_yticks([])

    ax11 = fig.add_subplot(spec[3, 1])
    ax11.imshow(seed[..., x_mid], cmap='magma', interpolation='None')
    ax11.axes.get_xaxis().set_visible(False)
    ax11.set_yticklabels([])
    ax11.set_yticks([])

    ax4 = fig.add_subplot(spec[0, 2])
    ax4.imshow(np.transpose(im[:, y_mid, ...]), cmap='magma', interpolation='None')
    ax4.set_yticklabels([])
    ax4.set_yticks([])
    ax4.set_xticklabels([])
    ax4.set_xticks([])
    ax4.set_xlabel('zx', fontdict=font)
    ax4.xaxis.set_label_position('top')

    ax5 = fig.add_subplot(spec[1, 2])
    ax5.imshow(np.transpose(gt_label[:, y_mid, ...]), cmap=new_cmp, interpolation='None')
    ax5.axes.get_xaxis().set_visible(False)
    ax5.set_yticklabels([])
    ax5.set_yticks([])

    ax6 = fig.add_subplot(spec[2, 2])
    ax6.imshow(np.transpose(pred_label[:, y_mid, ...]), cmap=new_cmp, interpolation='None')
    ax6.axes.get_xaxis().set_visible(False)
    ax6.set_yticklabels([])
    ax6.set_yticks([])

    ax7 = fig.add_subplot(spec[3, 2])
    ax7.imshow(np.transpose(seed[:, y_mid, ...]), cmap='magma', interpolation='None')
    ax7.axes.get_xaxis().set_visible(False)
    ax7.set_yticklabels([])
    ax7.set_yticks([])


    plt.tight_layout(pad=0, h_pad=0)
    plt.show()

def visualize_many_images(data_dir, project_name, train_val_dir, n_images, new_cmp):
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'bold',
            'size': 16,
            }
    im_filenames = sorted(glob(os.path.join(data_dir, project_name, train_val_dir) + '/images/*.tif'))
    ma_filenames = sorted(glob(os.path.join(data_dir, project_name, train_val_dir) + '/masks/*'))  # `tifs` or `csvs`
    indices = np.random.randint(0, len(im_filenames), n_images)
    fig = plt.figure(constrained_layout=False, figsize=(16, 10))
    spec = gridspec.GridSpec(ncols=n_images, nrows=3, figure=fig)
    for i, index in enumerate(indices):
        ax0 = fig.add_subplot(spec[0, i])
        im = tifffile.imread(im_filenames[index])
        if im.ndim == 2:
            ax0.imshow(im, cmap='magma', interpolation='None')
        else:
            ax0.imshow(im[0], cmap='magma', interpolation='None')
        ax0.axes.get_xaxis().set_visible(False)
        ax0.set_yticklabels([])
        ax0.set_yticks([])
        if i == 0:
            ax0.set_ylabel('IM', fontdict=font)
        ax1 = fig.add_subplot(spec[1, i])

        ma = tifffile.imread(ma_filenames[index])
        label, _, _ = relabel_sequential(ma)

        ax1.imshow(label, cmap=new_cmp, interpolation='None')
        ax1.axes.get_xaxis().set_visible(False)
        ax1.set_yticklabels([])
        ax1.set_yticks([])
        if i == 0:
            ax1.set_ylabel('INSTANCE MASK', fontdict=font)
        b = find_boundaries(label, mode='outer')
        res = (label > 0).astype(np.uint8)
        res[b] = 2
        ax2 = fig.add_subplot(spec[2, i])
        ax2.imshow(res, cmap='magma', interpolation='None')
        ax2.axes.get_xaxis().set_visible(False)
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        if i == 0:
            ax2.set_ylabel('SEMANTIC MASK', fontdict=font)

        plt.tight_layout(pad=0, h_pad=0)
    plt.show()

def visualize_many_volumes(data_dir, project_name, train_val_dir, new_cmp, anisotropy, index = None):
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'bold',
            'size': 16,
            }
    im_filenames = sorted(glob(os.path.join(data_dir, project_name, train_val_dir) + '/images/*.tif'))
    ma_filenames = sorted(glob(os.path.join(data_dir, project_name, train_val_dir) + '/masks/*.tif'))

    if index is None:
        index = np.random.randint(0, len(im_filenames), 1)[0]
    fig = plt.figure(constrained_layout=False, figsize=(12, 10))

    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
    im = tifffile.imread(im_filenames[index])
    label, _, _ = relabel_sequential(tifffile.imread(ma_filenames[index]))


    im = zoom(im, (anisotropy, 1, 1), order=0)
    # print(np.unique(label))
    label = zoom(label, (anisotropy, 1, 1), order=0)
    # print(np.unique(label))


    z_mid = im.shape[0] // 2
    y_mid = im.shape[1] // 2
    x_mid = im.shape[2] // 2

    ax0 = fig.add_subplot(spec[0, 0])
    ax0.imshow(im[z_mid, ...], cmap='magma', interpolation='None')
    ax0.set_yticklabels([])
    ax0.set_yticks([])
    ax0.set_xticklabels([])
    ax0.set_xticks([])
    ax0.set_ylabel('IM', fontdict=font)
    ax0.set_xlabel('xy', fontdict=font)
    ax0.xaxis.set_label_position('top')

    ax1 = fig.add_subplot(spec[1, 0])
    ax1.imshow(label[z_mid, ...], cmap=new_cmp, interpolation='None')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax1.set_ylabel('MASK', fontdict=font)



    # use interpolation
    ax6 = fig.add_subplot(spec[0, 1])
    ax6.imshow(im[..., x_mid], cmap='magma', interpolation='None')
    ax6.set_yticklabels([])
    ax6.set_yticks([])
    ax6.set_xticklabels([])
    ax6.set_xticks([])
    ax6.set_xlabel('yz', fontdict=font)
    ax6.xaxis.set_label_position('top')

    ax7 = fig.add_subplot(spec[1, 1])
    ax7.imshow(label[..., x_mid], cmap=new_cmp, interpolation='None')
    ax7.axes.get_xaxis().set_visible(False)
    ax7.set_yticklabels([])
    ax7.set_yticks([])



    ax3 = fig.add_subplot(spec[0, 2])
    ax3.imshow(np.transpose(im[:, y_mid, ...]), cmap='magma', interpolation='None')
    ax3.set_yticklabels([])
    ax3.set_yticks([])
    ax3.set_xticklabels([])
    ax3.set_xticks([])
    ax3.set_xlabel('zx', fontdict=font)
    ax3.xaxis.set_label_position('top')

    ax4 = fig.add_subplot(spec[1, 2])
    ax4.imshow(np.transpose(label[:, y_mid, ...]), cmap=new_cmp, interpolation='None')
    ax4.axes.get_xaxis().set_visible(False)
    ax4.set_yticklabels([])
    ax4.set_yticks([])


    plt.tight_layout(pad=0, h_pad=0)
    plt.show()