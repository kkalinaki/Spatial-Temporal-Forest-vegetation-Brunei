# this file helps in generating patches from the clip raster images that measure 2688 by 1792 pixels


import os
import cv2
import numpy as np
from patchify import patchify
from PIL import Image

root_directory = 'Datasets/'

patch_size = 128

img_dir=root_directory+"Raw Images/images/"
for path, subdirs, files in os.walk(img_dir):
    dirname = path.split(os.path.sep)[-1]
    images = os.listdir(path)  #List of all image names in this subdirectory
    for i, image_name in enumerate(images):  
        if image_name.endswith(".tif"):
            image = cv2.imread(path+"/"+image_name, 1)  #Read each image as BGR
            SIZE_X = (image.shape[1]//patch_size)*patch_size
            SIZE_Y = (image.shape[0]//patch_size)*patch_size 
            image = Image.fromarray(image)
            image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
            image = np.array(image)             
   
            #Extract patches from each image
            print("Now patchifying image:", path+"/"+image_name)
            patches_img = patchify(image, (128, 128, 3), step=128)  #Step=128 for 128 patches means no overlap
    
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    
                    single_patch_img = patches_img[i,j,:,:]
                    
                    single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                    
                    cv2.imwrite(root_directory+"128_patches/images/"+
                               image_name+"patch_"+str(i)+str(j)+".tif", single_patch_img)
                
            
  
 #Now do the same as above for masks
mask_dir=root_directory+"Raw Images/masks/"
for path, subdirs, files in os.walk(mask_dir):  
    dirname = path.split(os.path.sep)[-1]
    masks = os.listdir(path)  #List of all image names in this subdirectory
    for i, mask_name in enumerate(masks):  
        if mask_name.endswith(".tiff"):           
            mask = cv2.imread(path+"/"+mask_name, 0)  #Read each image as Grey (or color but remember to map each color to an integer)
            SIZE_X = (mask.shape[1]//patch_size)*patch_size 
            SIZE_Y = (mask.shape[0]//patch_size)*patch_size 
            mask = Image.fromarray(mask)
            mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
            mask = np.array(mask)             
   
            #Extract patches from each image
            print("Now patchifying mask:", path+"/"+mask_name)
            patches_mask = patchify(mask, (128, 128), step=128)  #Step=128 for 128 patches means no overlap
    
            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    
                    single_patch_mask = patches_mask[i,j,:,:]
                                         
                    cv2.imwrite(root_directory+"128_patches/masks/"+
                               mask_name+"patch_"+str(i)+str(j)+".tif", single_patch_mask)
                    