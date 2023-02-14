# Spatial-Temporal Mapping of Forest Vegetation Cover Changes Along Highways in Brunei Using Deep Learning Techniques and Sentinel-2 Images  
This is the repository for the experiment done to map forest cover changes along the Telisai-Lumut highway whose article has been submitted for publication.
The datasets folder contains two folders; Raw images folder contains images and their corresponding masks and 128_pacthes folder containing the patched images and masks for deep learning training.
# The create-patches file
This file contains the python code that shows how the patches wewre generated for network training
# Unet_backbones_and_ensemble.py
This is the python code for running the U-Net and its pretrained backbones and ensemble
# deeplabv3_plus.py
This is the python code for running the Deeplabv3+
# multi_class_u_net_model
This is the code for the standard U-Net model
