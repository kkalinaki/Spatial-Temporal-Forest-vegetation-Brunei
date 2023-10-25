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
# multi_class_u_net_model_training
This is the code for training the standard U-Net model


# Cite our paper using the following:
@article{kalinaki2023spatial,
  title={Spatial-temporal mapping of forest vegetation cover changes along highways in Brunei using deep learning techniques and Sentinel-2 images},
  author={Kalinaki, Kassim and Malik, Owais Ahmed and Lai, Daphne Teck Ching and Sukri, Rahayu Sukmaria and Wahab, Rodzay Bin Haji Abdul},
  journal={Ecological Informatics},
  volume={77},
  pages={102193},
  year={2023},
  publisher={Elsevier}
}
