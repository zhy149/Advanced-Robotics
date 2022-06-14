Files in this directories: train.py, test.py, utils_depth.py, utils_rgb.py, utils_rgbd.py, please keep all these files with the same directories with the three models folder containing three models, along with the dataset folder containing frames.npy.
Note that the models and dataset are too large to upload, therefore they are left on deathstar.
The dataset folder is in the directory /media/rpal/SSD512/Zihe/adv_robo_proj, and the three models folder are there too. All other codes necessary in this folder are in the directory as well for convenience.

To run the code:

For training:

python train.py [rgb\depth\rgbd], choose one of the three options to start training, the best model will be saved as checkpoint.h5 in the same directory.

For testing:

python test.py [rgb\depth\rgbd], choose one of the three options to start testing, the test.py will load the corresponding model in the threemodels folder.

A folder either empty or not empty called results is included in this folder, make sure there is one and only one such folder in this directory to store training and testing result.