##### This file combines all data - training, testing and validation
##### It also applies all the backbones and ensemble after splitting data
import os
path = "Datasets"
os.chdir(path)

import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras 

from tensorflow.keras.metrics import MeanIoU

SIZE_X = 128 
SIZE_Y = 128
n_classes=4 #Number of classes for segmentation

#Capture training image info as a list
train_images = []
name_of_images=[]
for directory_path in glob.glob("/128_patches/images/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        name_of_images.append(img_path[19:-4])
        img = cv2.imread(img_path, 1)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
       
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = [] 
for directory_path in glob.glob("/128_patches/masks/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)
        train_masks.append(mask)
        
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)

#Encode labels and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

#################################################
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#Create a subset of data for quick testing
#Picking 15% for testing and remaining for training
from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.15, random_state = 0)


train_name_of_images=name_of_images[0:len(X1)]
test_name_of_images=name_of_images[len(X1):]

#Further split training data t a smaller subset for quick testing of models
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.15, random_state = 0)

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

from tensorflow.keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

#Reused parameters in all models

n_classes=4
activation='softmax'

LR = 0.0001
optim = keras.optimizers.Adam(LR)

# set class weights for dice_loss 
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# define metrics and threshold
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

########################################################################
###Model 1
BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_train1 = preprocess_input1(X_train)
X_test1 = preprocess_input1(X_test)

# define model
model1 = sm.Unet(BACKBONE1, encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
model1.compile(optim, total_loss, metrics=metrics)

print(model1.summary())

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='resnet34_logs')]

history1=model1.fit(X_train1, 
          y_train_cat,
          batch_size=4, 
          epochs=100,
          verbose=1,
          validation_data=(X_test1, y_test_cat))

model1.save('res34_backbone_100epochs_alldata.hdf5')

#plot the training and validation accuracy and loss at each epoch
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history1.history['iou_score']
val_acc = history1.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()
############################################################

###Model 2
BACKBONE2 = 'inceptionv3'
preprocess_input2 = sm.get_preprocessing(BACKBONE2)

# preprocess input
X_train2 = preprocess_input2(X_train)
X_test2 = preprocess_input2(X_test)

# define model
model2 = sm.Unet(BACKBONE2, encoder_weights='imagenet', classes=n_classes, activation=activation)


# compile keras model with defined optimozer, loss and metrics
model2.compile(optim, total_loss, metrics)

print(model2.summary())

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='inceptionv3_logs')]

history2=model2.fit(X_train2, 
          y_train_cat,
          batch_size=4, 
          epochs=100,
          verbose=1,
          validation_data=(X_test2, y_test_cat))


model2.save('inceptionv3_backbone_100epochs_alldata.hdf5')

#plot the training and validation accuracy and loss at each epoch
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history2.history['iou_score']
val_acc = history2.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()

#####################################################
###Model 3

BACKBONE3 = 'vgg19'
preprocess_input3 = sm.get_preprocessing(BACKBONE3)

# preprocess input
X_train3 = preprocess_input3(X_train)
X_test3 = preprocess_input3(X_test)


# define model
model3 = sm.Unet(BACKBONE3, encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
model3.compile(optim, total_loss, metrics)

print(model3.summary())

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='vgg19_logs')]

history3=model3.fit(X_train3, 
          y_train_cat,
          batch_size=4, 
          epochs=100,
          verbose=1,
          validation_data=(X_test3, y_test_cat))


model3.save('vgg19_backbone_100epochs_alldata.hdf5')

#plot the training and validation accuracy and loss at each epoch
loss = history3.history['loss']
val_loss = history3.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history3.history['iou_score']
val_acc = history3.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()

########################################################################
###Model 4
BACKBONE4 = 'efficientnetb7'
preprocess_input4 = sm.get_preprocessing(BACKBONE4)

# preprocess input
X_train4 = preprocess_input4(X_train)
X_test4 = preprocess_input4(X_test)

# define model
model4 = sm.Unet(BACKBONE4, encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
model4.compile(optim, total_loss, metrics)

print(model4.summary())

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='efficientnetb7_logs')]


history4=model4.fit(X_train4, 
          y_train_cat,
          batch_size=4, 
          epochs=100,
          verbose=1,
          validation_data=(X_test4, y_test_cat))


model4.save('efficientnetb7_backbone_100epochs_alldata_new.hdf5')

#plot the training and validation accuracy and loss at each epoch
loss = history4.history['loss']
val_loss = history4.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history4.history['iou_score']
val_acc = history4.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()
#####################################################


from tensorflow.keras.models import load_model

### FOCUS ON A SINGLE MODEL

#Set compile=False as we are not loading it for training, only for prediction.
model1 = load_model('res34_backbone_100epochs_alldata.hdf5', compile=False)
model2 = load_model('inceptionv3_backbone_100epochs_alldata.hdf5', compile=False)
model3 = load_model('vgg19_backbone_100epochs_alldata.hdf5', compile=False)
model4 = load_model('efficientnetb7_backbone_100epochs_alldata_new.hdf5', compile=False)
#######################################
#IOU - model1
#######################################
#### Prediction on Unseen Data
######################################
X_do_not_use1 = preprocess_input1(X_do_not_use)
y_pred_do_not_use1=model1.predict(X_do_not_use1)
y_pred_do_not_use1_argmax=np.argmax(y_pred_do_not_use1, axis=3)

#Using built in keras function
#from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_do_not_use[:,:,:,0], y_pred_do_not_use1_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

#Test some random images

import random
for i in range(len(X_test1)):
    
    test_img = X_do_not_use[i]
    ground_truth=y_do_not_use[i]
    test_img_input=np.expand_dims(test_img, 0)
    test_img_input1 = preprocess_input1(test_img_input)
    test_pred1 = model1.predict(test_img_input1)
    test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]
    

    fig=plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(test_prediction1, cmap='jet')
    plt.show()
  

#### Overlaying Images
#######################################
plt.figure(figsize=(12, 8))
plt.subplot(1,2,1)
plt.imshow(test_img[:,:,0], 'gray', interpolation='none')
plt.subplot(1,2,2)
plt.imshow(test_img[:,:,0], 'gray', interpolation='none')
plt.imshow(test_prediction1, 'jet', interpolation='none', alpha=0.7)
plt.show()

# Prediction on Validation Data
y_pred1=model1.predict(X_test1)
y_pred1_argmax=np.argmax(y_pred1, axis=3)

#Using built in keras function
#from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred1_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

#Vaerify the prediction on first image
plt.imshow(train_images[0, :,:,0], cmap='gray')
plt.imshow(train_masks[0], cmap='gray')

#Test some random images
import random
test_img_number = random.randint(0, len(X_test1))
test_img = X_test1[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)

test_img_input1 = preprocess_input1(test_img_input)

test_pred1 = model1.predict(test_img_input1)
test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction1, cmap='jet')
plt.show()
#######################################


#IOU - model2
#######################################
#### Prediction on Unseen Data
######################################
X_do_not_use2 = preprocess_input2(X_do_not_use)
y_pred_do_not_use2=model2.predict(X_do_not_use2)
y_pred_do_not_use2_argmax=np.argmax(y_pred_do_not_use2, axis=3)

#Using built in keras function
#from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_do_not_use[:,:,:,0], y_pred_do_not_use2_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

import random
for i in range(len(X_test1)):
    
    test_img = X_do_not_use[i]
    ground_truth=y_do_not_use[i]
    test_img_input=np.expand_dims(test_img, 0)
    test_img_input2 = preprocess_input2(test_img_input)
    test_pred2 = model2.predict(test_img_input2)
    test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]
    

    fig=plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(test_prediction2, cmap='jet')
    plt.show()
  
# Overlaying Images
#######################################
plt.figure(figsize=(12, 8))
plt.subplot(1,2,1)
plt.imshow(test_img[:,:,0], 'gray', interpolation='none')
plt.subplot(1,2,2)
plt.imshow(test_img[:,:,0], 'gray', interpolation='none')
plt.imshow(test_prediction2, 'jet', interpolation='none', alpha=0.7)
plt.show()
#######################################
#### Prediction on Validation Data
######################################
y_pred2=model2.predict(X_test2)
y_pred2_argmax=np.argmax(y_pred2, axis=3)

#Using built in keras function
#from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred2_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

#Vaerify the prediction on first image
plt.imshow(train_images[0, :,:,0], cmap='gray')
plt.imshow(train_masks[0], cmap='gray')

#Test some random images
import random
test_img_number = random.randint(0, len(X_test2))
test_img = X_test2[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)

test_img_input2 = preprocess_input2(test_img_input)

test_pred2 = model2.predict(test_img_input2)
test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction2, cmap='jet')
plt.show()

#IOU - model3
## Prediction on Unseen Data
######################################
X_do_not_use3 = preprocess_input3(X_do_not_use)
y_pred_do_not_use3=model3.predict(X_do_not_use3)
y_pred_do_not_use3_argmax=np.argmax(y_pred_do_not_use3, axis=3)

#Using built in keras function
#from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_do_not_use[:,:,:,0], y_pred_do_not_use3_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

#Test some random images

import random
for i in range(len(X_test1)):
    
    test_img = X_do_not_use[i]
    ground_truth=y_do_not_use[i]
    test_img_input=np.expand_dims(test_img, 0)
    test_img_input3 = preprocess_input3(test_img_input)
    test_pred3 = model3.predict(test_img_input3)
    test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]
    fig=plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(test_prediction3, cmap='jet')
    plt.show()

## Overlaying Images
#######################################
plt.figure(figsize=(12, 8))
plt.subplot(1,2,1)
plt.imshow(test_img[:,:,0], 'gray', interpolation='none')
plt.subplot(1,2,2)
plt.imshow(test_img[:,:,0], 'gray', interpolation='none')
plt.imshow(test_prediction3, 'jet', interpolation='none', alpha=0.7)
plt.show()
#######################################
#### Prediction on Validation Data
######################################
y_pred3=model3.predict(X_test3)
y_pred3_argmax=np.argmax(y_pred3, axis=3)

#Using built in keras function
#from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred3_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

#Vaerify the prediction on first image
plt.imshow(train_images[0, :,:,0], cmap='gray')
plt.imshow(train_masks[0], cmap='gray')
##############################################################

#Test some random images
import random
test_img_number = random.randint(0, len(X_test3))
test_img = X_test3[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)

test_img_input3 = preprocess_input3(test_img_input)

test_pred3 = model3.predict(test_img_input3)
test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction3, cmap='jet')
plt.show()

#IOU - model4
#######################################
#### Prediction on Unseen Data
######################################
X_do_not_use4 = preprocess_input4(X_do_not_use)
y_pred_do_not_use4=model4.predict(X_do_not_use4)
y_pred_do_not_use4_argmax=np.argmax(y_pred_do_not_use4, axis=3)


#Using built in keras function
#from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_do_not_use[:,:,:,0], y_pred_do_not_use4_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

import random
for i in range(len(X_test1)):
    
    test_img = X_do_not_use[i]
    ground_truth=y_do_not_use[i]
    test_img_input=np.expand_dims(test_img, 0)
    test_img_input4 = preprocess_input4(test_img_input)
    test_pred4 = model4.predict(test_img_input4)
    test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]
    fig=plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(test_prediction4, cmap='jet')
    plt.show()

####################################### 
#### Overlaying Images
#######################################
plt.figure(figsize=(12, 8))
plt.subplot(1,2,1)
plt.imshow(test_img[:,:,0], 'gray', interpolation='none')
plt.subplot(1,2,2)
plt.imshow(test_img[:,:,0], 'gray', interpolation='none')
plt.imshow(test_prediction4, 'jet', interpolation='none', alpha=0.7)
plt.show()
#######################################
#### Prediction on Validation Data
######################################
y_pred4=model4.predict(X_test4)
y_pred4_argmax=np.argmax(y_pred4, axis=3)


#Using built in keras function
#from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred4_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

#Vaerify the prediction on first image
plt.imshow(train_images[0, :,:,0], cmap='gray')
plt.imshow(train_masks[0], cmap='gray')
##############################################################

#Test some random images
import random
test_img_number = random.randint(0, len(X_test4))
test_img = X_test4[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)

test_img_input4 = preprocess_input4(test_img_input)

test_pred4 = model4.predict(test_img_input4)
test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction4, cmap='jet')
plt.show()


#### Ensemble Models
################################################################


models = [model1, model2, model3, model4]

#############################################
#Prediction on Unseen Data
#############################################
pred_do_not_use1 = model1.predict(X_do_not_use1)
pred_do_not_use2 = model2.predict(X_do_not_use2)
pred_do_not_use3 = model3.predict(X_do_not_use3)
pred_do_not_use4 = model4.predict(X_do_not_use4)

preds=np.array([pred_do_not_use1, pred_do_not_use2, pred_do_not_use3, pred_do_not_use4])

#preds=np.array(preds)
weights = [0.25, 0.25, 0.25, 0.25]
#weights = [0.3, 0.1, 0.2, 0.3]
#Use tensordot to sum the products of all elements over specified axes.
weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
weighted_ensemble_prediction = np.argmax(weighted_preds, axis=3)

y_pred1_argmax=np.argmax(pred_do_not_use1, axis=3)
y_pred2_argmax=np.argmax(pred_do_not_use2, axis=3)
y_pred3_argmax=np.argmax(pred_do_not_use3, axis=3)
y_pred4_argmax=np.argmax(pred_do_not_use4, axis=3)

#Using built in keras function
n_classes = 4
IOU1 = MeanIoU(num_classes=n_classes)  
IOU2 = MeanIoU(num_classes=n_classes)  
IOU3 = MeanIoU(num_classes=n_classes)  
IOU4 = MeanIoU(num_classes=n_classes)  
IOU_weighted = MeanIoU(num_classes=n_classes)  

IOU1.update_state(y_do_not_use[:,:,:,0], y_pred1_argmax)
IOU2.update_state(y_do_not_use[:,:,:,0], y_pred2_argmax)
IOU3.update_state(y_do_not_use[:,:,:,0], y_pred3_argmax)
IOU4.update_state(y_do_not_use[:,:,:,0], y_pred4_argmax)
IOU_weighted.update_state(y_do_not_use[:,:,:,0], weighted_ensemble_prediction)


print('IOU Score for model1 = ', IOU1.result().numpy())
print('IOU Score for model2 = ', IOU2.result().numpy())
print('IOU Score for model3 = ', IOU3.result().numpy())
print('IOU Score for model4 = ', IOU4.result().numpy())
print('IOU Score for weighted average ensemble = ', IOU_weighted.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_weighted.get_weights()).reshape(n_classes, n_classes)
print(values)

class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for individual classes using ensemble")
print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)
###########################################
#Grid search for the best combination of w1, w2, w3 that gives maximum acuracy

import pandas as pd
df = pd.DataFrame([])

for w1 in range(0, 4):
    for w2 in range(0,4):
        for w3 in range(0,4):
            for w4 in range(0,4):
                wts = [w1/10.,w2/10.,w3/10.,w4/10.]
                
                IOU_wted = MeanIoU(num_classes=n_classes) 
                wted_preds = np.tensordot(preds, wts, axes=((0),(0)))
                wted_ensemble_pred = np.argmax(wted_preds, axis=3)
                IOU_wted.update_state(y_do_not_use[:,:,:,0], wted_ensemble_pred)
                print("Now predciting for weights :", w1/10., w2/10., w3/10., w4/10.," : IOU = ", IOU_wted.result().numpy())
                df = df.append(pd.DataFrame({'wt1':wts[0],'wt2':wts[1], 
                                             'wt3':wts[2], 'wt4':wts[3], 'IOU': IOU_wted.result().numpy()}, index=[0]), ignore_index=True)
            
max_iou_row = df.iloc[df['IOU'].idxmax()]
print("Max IOU of ", max_iou_row[3], " obained with w1=", max_iou_row[0],
      " w2=", max_iou_row[1], " w3=", max_iou_row[2], " and w4=", max_iou_row[3])         


#############################################################
opt_weights = [max_iou_row[0], max_iou_row[1], max_iou_row[2], max_iou_row[3]]

#Use tensordot to sum the products of all elements over specified axes.
opt_weighted_preds = np.tensordot(preds, opt_weights, axes=((0),(0)))
opt_weighted_ensemble_prediction = np.argmax(opt_weighted_preds, axis=3)

#############################################
#Prediction on Validation Data
#############################################
pred1 = model1.predict(X_test1)
pred2 = model2.predict(X_test2)
pred3 = model3.predict(X_test3)
pred4 = model4.predict(X_test4)

preds=np.array([pred1, pred2, pred3, pred4])

#preds=np.array(preds)
weights = [0.25, 0.25, 0.25, 0.25]

#Use tensordot to sum the products of all elements over specified axes.
weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
weighted_ensemble_prediction = np.argmax(weighted_preds, axis=3)

y_pred1_argmax=np.argmax(pred1, axis=3)
y_pred2_argmax=np.argmax(pred2, axis=3)
y_pred3_argmax=np.argmax(pred3, axis=3)
y_pred4_argmax=np.argmax(pred4, axis=3)

#Using built in keras function
n_classes = 4
IOU1 = MeanIoU(num_classes=n_classes)  
IOU2 = MeanIoU(num_classes=n_classes)  
IOU3 = MeanIoU(num_classes=n_classes)  
IOU4 = MeanIoU(num_classes=n_classes)  
IOU_weighted = MeanIoU(num_classes=n_classes)  

IOU1.update_state(y_test[:,:,:,0], y_pred1_argmax)
IOU2.update_state(y_test[:,:,:,0], y_pred2_argmax)
IOU3.update_state(y_test[:,:,:,0], y_pred3_argmax)
IOU4.update_state(y_test[:,:,:,0], y_pred4_argmax)
IOU_weighted.update_state(y_test[:,:,:,0], weighted_ensemble_prediction)


print('IOU Score for model1 = ', IOU1.result().numpy())
print('IOU Score for model2 = ', IOU2.result().numpy())
print('IOU Score for model3 = ', IOU3.result().numpy())
print('IOU Score for model4 = ', IOU4.result().numpy())
print('IOU Score for weighted average ensemble = ', IOU_weighted.result().numpy())

#To calculate I0U for each class...
values = np.array(IOU_weighted.get_weights()).reshape(n_classes, n_classes)
print(values)

class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for individual classes using ensemble")
print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class3_IoU)
###########################################
#Grid search for the best combination of w1, w2, w3 that gives maximum acuracy

import pandas as pd
df = pd.DataFrame([])

for w1 in range(0, 4):
    for w2 in range(0,4):
        for w3 in range(0,4):
            for w4 in range(0,4):
                wts = [w1/10.,w2/10.,w3/10.,w4/10.]
                
                IOU_wted = MeanIoU(num_classes=n_classes) 
                wted_preds = np.tensordot(preds, wts, axes=((0),(0)))
                wted_ensemble_pred = np.argmax(wted_preds, axis=3)
                IOU_wted.update_state(y_test[:,:,:,0], wted_ensemble_pred)
                print("Now predciting for weights :", w1/10., w2/10., w3/10., w4/10.," : IOU = ", IOU_wted.result().numpy())
                df = df.append(pd.DataFrame({'wt1':wts[0],'wt2':wts[1], 
                                             'wt3':wts[2], 'wt4':wts[3], 'IOU': IOU_wted.result().numpy()}, index=[0]), ignore_index=True)
            
max_iou_row = df.iloc[df['IOU'].idxmax()]
print("Max IOU of ", max_iou_row[3], " obained with w1=", max_iou_row[0],
      " w2=", max_iou_row[1], " w3=", max_iou_row[2], " and w4=", max_iou_row[3])         


#############################################################
opt_weights = [max_iou_row[0], max_iou_row[1], max_iou_row[2], max_iou_row[3]]

#Use tensordot to sum the products of all elements over specified axes.
opt_weighted_preds = np.tensordot(preds, opt_weights, axes=((0),(0)))
opt_weighted_ensemble_prediction = np.argmax(opt_weighted_preds, axis=3)
#######################################################
#Predict on a few images
#######################################################

import random

for i in range(len(X_test1)):
    
    test_img = X_do_not_use[i]
    ground_truth=y_do_not_use[i]
    test_img_norm=test_img[:,:,:]
    test_img_input=np.expand_dims(test_img_norm, 0)

    #Weighted average ensemble
    models = [model1, model2, model3, model4]
    
    test_img_input1 = preprocess_input1(test_img_input)
    test_img_input2 = preprocess_input2(test_img_input)
    test_img_input3 = preprocess_input3(test_img_input)
    test_img_input4 = preprocess_input4(test_img_input)
    
    test_pred1 = model1.predict(test_img_input1)
    test_pred2 = model2.predict(test_img_input2)
    test_pred3 = model3.predict(test_img_input3)
    test_pred4 = model4.predict(test_img_input4)
    
    test_preds=np.array([test_pred1, test_pred2, test_pred3, test_pred4])
    
    #Use tensordot to sum the products of all elements over specified axes.
    weighted_test_preds = np.tensordot(test_preds, opt_weights, axes=((0),(0)))
    weighted_ensemble_test_prediction = np.argmax(weighted_test_preds, axis=3)[0,:,:]
    
    
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(weighted_ensemble_test_prediction, cmap='jet')
    plt.savefig(r'D:\Kassim\Lumut7\Datasets\sample_predictions_ensemble\%s'%test_name_of_images[i][26:]+'.png',dpi=300)
    plt.show()

#### Overlaying Images
plt.figure(figsize=(12, 8))
plt.subplot(1,2,1)
plt.imshow(test_img[:,:,0], 'gray', interpolation='none')
plt.subplot(1,2,2)
plt.imshow(test_img[:,:,0], 'gray', interpolation='none')
plt.imshow(weighted_ensemble_test_prediction, 'jet', interpolation='none', alpha=0.7)
plt.show()

#### Prediction on validation data
import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,:]
test_img_input=np.expand_dims(test_img_norm, 0)

#Weighted average ensemble
models = [model1, model2, model3, model4]

test_img_input1 = preprocess_input1(test_img_input)
test_img_input2 = preprocess_input2(test_img_input)
test_img_input3 = preprocess_input3(test_img_input)
test_img_input4 = preprocess_input4(test_img_input)

test_pred1 = model1.predict(test_img_input1)
test_pred2 = model2.predict(test_img_input2)
test_pred3 = model3.predict(test_img_input3)
test_pred4 = model4.predict(test_img_input4)

test_preds=np.array([test_pred1, test_pred2, test_pred3, test_pred4])

#Use tensordot to sum the products of all elements over specified axes.
weighted_test_preds = np.tensordot(test_preds, opt_weights, axes=((0),(0)))
weighted_ensemble_test_prediction = np.argmax(weighted_test_preds, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(weighted_ensemble_test_prediction, cmap='jet')
plt.show()