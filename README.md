# Exercise 3: Pixel Classification (Segmentation) with an U-Net

## Connect to your HT Jupyter instance...

As we did likely in the connection tutorial, you need to:

1. SSH into our cluster to enable port forwarding. The command is something like:

```
ssh your.user@hpclogin.fht.org -L 8888:gnodeXX:YYYY -L ZZZZ:gnodeXX:ZZZZ
```
Where you obviously will have to replace XX and YYYY, as well as insert your real user name. Note that ZZZZ is your TensorBoard port and that today you will have to
use ZZZZ locally and remotely (no worries you can fix that later).

2. Now connect to your Jupyter instance from your local broser by going to:
```
localhost:8888
```

## Clone this repo...
In Jupyter...
* Open a terminal window (inside the browser, from within Jupyter).
* Clone this repository by writing `git clone https://github.com/dl4mia/03_segmentation_unet.git`.
* A new folder was created, containing all three exercises. In order to run them we need to first create a stutable conda environment.

## Setup conda...

From within the same terminal in your browser, create a `conda` environment for this exercise, activate it, and install a bunch of things this exercise needs:

```
conda create -n 03_segmentation_unet python=3.7
conda activate 03_segmentation_unet
conda install -c pytorch pytorch torchvision cudatoolkit=10.2 jupyter
pip install tensorboard imageio scipy
```

Now navigate to the exercise folder using jupyter and get going! :)
