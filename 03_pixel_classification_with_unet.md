# Pixel Classification with U-Net (DL4MIA-22)

### Goals

Train and Infer with a 2D *U-Net* network

### Overview

In this tutorial, we shall learn to use *pixel classification* **notebooks to train a model for performing semantic segmentation on 2D** microscopy images.

We shall train a model to map each pixel to three values - the probability of it belonging to the foreground class, the probability of it belonging to the membrane class and the probability of it belonging to the background class.

Since the pixels belonging to the membrane class are typically fewer than the other two classes, we shall weigh their opinion more while computing the loss.

In order to ensure that the total probability for any pixel sums to 1 and that each predicted class probability is a positive value, we shall use a `softmax` activation as the final layer of the network.

General pipeline of each notebook is:

- **Preprocess data**

We calculate some properties about the data, for example, what is the typical smallest object size in the training label masks.

- **Train network for a few epochs**

We learn to initialize the network, the train and val datasets and data-loaders and then update the model weights for a few epochs.

- **Predict on evaluation data**

We test the trained model on data which has never been shown to the model during the training phase.

Also, since we are often interested in each object being assigned a unique id (i.e. the task of *instance segmentation*), we perform *connected component analysis* in order to get unique ids for each object. We compute the mean accuracy score (between `0` and `1.0` where `1.0` indicates a perfect prediction) which tells us the quality of our predicted instance masks *vis-a-vis* the ground truth instance masks.

### [0/1] Download packages

Open a fresh terminal window (click on *Activities* at top-left and select *Terminal* Window) , change directory to `DL4MIA` and enter the following commands:

```bash
cd DL4MIA
conda create -y -n pytorchEnv python==3.7
conda activate pytorchEnv

conda install cudatoolkit=11.3 -c conda-forge
pip install torch torchvision

git clone https://github.com/dl4mia/03_segmentation_unet.git
cd 03_segmentation_unet
pip install -e .

python3 -c "import torch; print(torch.cuda.is_available())
```

### [1/1] **Train and Infer with a 2D *U-Net* network**

Browse to the 2D exercise:

```bash
cd 03_segmentation_unet/examples/DL4MIA/train-and-predict-2d.ipynb
jupyter notebook
```

Total runtime of this notebook (`Kernel >Restart and Run All`) is around 20 min.