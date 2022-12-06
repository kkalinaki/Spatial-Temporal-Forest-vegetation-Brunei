import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import albumentations as A
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
path = "Datasets"
os.chdir(path)

os.listdir(os.path.join('128_patches', 'masks'))[:924]
os.listdir(os.path.join('128_patches', 'images'))[:924]

config = {
    'IMG_PATH': os.path.join('128_patches', 'images'),
    'LABEL_PATH': os.path.join('128_patches', 'masks'),
    'NUM_CLASSES': 4,
    'BATCH_SIZE': 4,
    'IMAGE_SIZE': 128
}
##### Building Dataset
image_paths =  glob(os.path.join(config['IMG_PATH'], '*'), recursive=True)
mask_paths =  glob(os.path.join(config['LABEL_PATH'], '*'), recursive=True)

image_paths_train1, image_paths_test1, mask_paths_train1, mask_paths_test1 = train_test_split(image_paths, mask_paths, test_size=0.15)
image_paths_train, image_paths_test, mask_paths_train, mask_paths_test = train_test_split(image_paths_train1, mask_paths_train1, test_size=0.15)

config['DATASET_LENGTH'] = len(image_paths_train)

def preprocess(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size=[config['IMAGE_SIZE'], config['IMAGE_SIZE']])
    img = tf.cast(img, tf.float32) / 255.0
    
    mask = tf.io.read_file(mask_path)
    # Only one channel for masks, denoting the class and NOT image colors
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, size=[config['IMAGE_SIZE'], config['IMAGE_SIZE']])
    mask = tf.cast(mask, tf.float32)
    return img, mask

def augment_dataset_tf(img, mask):
     #  Augmentations should always be performed on both an input image and a mask if applied at all
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.rot90(img)
        mask = tf.image.rot90(mask)
            
    return img, mask

def albumentations(img, mask):
    # Augmentation pipeline - each of these has an adjustable probability
    # of being applied, regardless of other transforms
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=70),
        # CoarseDropout is the new Cutout implementation
        A.CoarseDropout(p=0.5, max_holes=12, max_height=24, max_width=24)
    ])
    
    # Apply transforms and extract image and mask
    transformed = transform(image=img, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']
    
    # Cast to TF Floats and return
    transformed_image = tf.cast(transformed_image, tf.float32)
    transformed_mask = tf.cast(transformed_mask, tf.float32)
    return transformed_image, transformed_mask

def create_dataset_tf(images, masks, augment):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks)).shuffle(len(images))
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.map(apply_albumentations, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(config['BATCH_SIZE'], drop_remainder=True).prefetch(tf.data.AUTOTUNE).repeat()
    else:
        dataset = dataset.batch(config['BATCH_SIZE'], drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def apply_albumentations(img, mask):
    aug_img, aug_mask = tf.numpy_function(func=albumentations, inp=[img, mask], Tout=[tf.float32, tf.float32])
    aug_img = tf.ensure_shape(aug_img, shape=[config['IMAGE_SIZE'], config['IMAGE_SIZE'], 3])
    aug_mask = tf.ensure_shape(aug_mask, shape=[config['IMAGE_SIZE'], config['IMAGE_SIZE'], 1])
    return aug_img, aug_mask

train_set = create_dataset_tf(image_paths_train, mask_paths_train, augment=False)
test_set = create_dataset_tf(image_paths_test, mask_paths_test, augment=False)

for img_batch, mask_batch in train_set.take(2):
    for i in range(len(img_batch)):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img_batch[i].numpy())
        ax[1].imshow(mask_batch[i].numpy())
        
# Turns into atrous_block with dilation_rate > 1
def conv_block(block_input, num_filters=256, kernel_size=(3, 3), dilation_rate=1, padding="same"):
    x = keras.layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same")(block_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x

# Atrous Spatial Pyramid Pooling
def ASPP(inputs):
    # 4 conv blocks with dilation rates at `[1, 6, 12, 18]`
    conv_1 = conv_block(inputs, kernel_size=(1, 1), dilation_rate=1)
    conv_6 = conv_block(inputs, kernel_size=(3, 3), dilation_rate=6)
    conv_12 = conv_block(inputs, kernel_size=(3, 3), dilation_rate=12)
    conv_18 = conv_block(inputs, kernel_size=(3, 3), dilation_rate=18)
    
    dims = inputs.shape
    x = keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(inputs)
    x = conv_block(x, kernel_size=1)
    out_pool = keras.layers.UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]))(x)

    x = keras.layers.Concatenate()([conv_1, conv_6, conv_12, conv_18, out_pool])
    return conv_block(x, kernel_size=1)

def define_deeplabv3_plus(image_size, num_classes, backbone):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    
    if backbone == 'resnet':
        resnet101 = keras.applications.ResNet152(
            weights="imagenet", 
            include_top=False, 
            input_tensor=model_input)
        x = resnet101.get_layer("conv4_block6_2_relu").output
        low_level = resnet101.get_layer("conv2_block3_2_relu").output
        
    elif backbone == 'effnet':
        effnet = keras.applications.EfficientNetV2B1(
            weights="imagenet", 
             include_top=False, 
             input_tensor=model_input)
        x = effnet.get_layer("block5e_activation").output
        low_level = effnet.get_layer("block2a_expand_activation").output
        
    aspp_result = ASPP(x)
    upsampled_aspp = keras.layers.UpSampling2D(size=(4, 4))(aspp_result)
    
    low_level = conv_block(low_level, num_filters=48, kernel_size=1)

    x = keras.layers.Concatenate()([upsampled_aspp, low_level])
    x = conv_block(x)
    x = keras.layers.UpSampling2D(size=(4, 4))(x)
    model_output = keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation='softmax')(x)
    return keras.Model(inputs=model_input, outputs=model_output)

model = define_deeplabv3_plus(config['IMAGE_SIZE'], config['NUM_CLASSES'], 'effnet')
model.summary()

from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=config['NUM_CLASSES'])[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

class MeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
        super(MeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
    
reduceLr = keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.3, monitor='val_sparse_categorical_accuracy')
early_stopping = keras.callbacks.EarlyStopping(patience=10, monitor='val_sparse_categorical_accuracy', restore_best_weights=True)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    #loss=soft_dice_loss,
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy", 
              MeanIoU(num_classes=config['NUM_CLASSES']),
              dice_coef])

history = model.fit(train_set, 
                    epochs=100, 
                    steps_per_epoch=int(config['DATASET_LENGTH']/config['BATCH_SIZE']), 
                    validation_data=test_set,
                    callbacks=[reduceLr, early_stopping])

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
dice_coeff = history.history['dice_coef']
val_dice_coeff = history.history['val_dice_coef']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')

plt.title('Training and validation performance')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save('deeplabv3plus_multi_class_lre4123.hdf5')

from tensorflow.keras.models import load_model
model = load_model('deeplabv3plus_multi_class_lre4123.hdf5', compile=False)
########Visualization

def visualize_predictions(img_num):
    if os.path.exists(os.path.join(config['IMG_PATH'], f'{img_num}.jpg')):
        img = cv2.imread(os.path.join(config['IMG_PATH'], f'{img_num}.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,  [config['IMAGE_SIZE'], config['IMAGE_SIZE']])
        img = img/255.0

        mask = cv2.imread(os.path.join(config['LABEL_PATH'], f'{img_num}.png'))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask,  [config['IMAGE_SIZE'], config['IMAGE_SIZE']])
               
        pred = model.predict(np.expand_dims(img, 0), verbose=0)
        predictions = np.argmax(pred, axis=-1)
        fig, ax = plt.subplots(1, 4, figsize=(16, 8))
        ax[0].imshow(img)
        ax[1].imshow(mask)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[3].axis('off')
        ax[0].set_title('Image')
        ax[1].set_title('Mask')
        ax[3].set_title('Prediction')
        plt.show()

visualize_predictions('2')
visualize_predictions('35')
visualize_predictions('38')

######## Predict and overlay
def predict_and_overlay(img_num):
    if os.path.exists(os.path.join(config['IMG_PATH'], f'{img_num}.jpg')):

        img = cv2.imread(os.path.join(config['IMG_PATH'], f'{img_num}.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,  [config['IMAGE_SIZE'], config['IMAGE_SIZE']])
        img = img/255.0
        pred = model.predict(np.expand_dims(img, 0), verbose=0)
        predictions = np.argmax(pred, axis=-1)
        fig, ax = plt.subplots(1, figsize=(16, 8))
        ax.imshow(img)
        ax.axis('off')
        plt.show()
       
predict_and_overlay(38)        


########Compute IOU-MIOU
def predict_image(img_num):
    if os.path.exists(os.path.join(config['IMG_PATH'], f'{img_num}.jpg')):

        img = cv2.imread(os.path.join(config['IMG_PATH'], f'{img_num}.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,  [config['IMAGE_SIZE'], config['IMAGE_SIZE']])
        img = img/255.0

        pred = model.predict(np.expand_dims(img, 0), verbose=0)
        predictions = np.argmax(pred, axis=-1)
        return predictions

from pathlib import Path
from tensorflow.keras.metrics import MeanIoU

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
miou = []
class1_IoU = []
class2_IoU = []
class3_IoU = []
class4_IoU = []

# 

for i in range(len(image_paths_test1)):
    path = Path(image_paths_test1[i])
    filename = os.path.basename(path)
    outfile = filename.split('.')[0]
    prediction_mask = predict_image(outfile)
    image_org = cv2.imread(image_paths_test1[i],0)
    img_org = cv2.resize(image_org,(config['IMAGE_SIZE'],config['IMAGE_SIZE']))
    image_masks = cv2.imread(mask_paths_test1[i],0)
    img = cv2.resize(image_masks,(config['IMAGE_SIZE'],config['IMAGE_SIZE']))
    IOU_keras.update_state(img, prediction_mask)
    miou.append(IOU_keras.result().numpy())
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[0,5] + values[0,6] + values[0,7] + values[0,8] + values[1,0]+ values[2,0]+ values[3,0]+ values[4,0]+ values[5,0]+ values[6,0]+ values[7,0]+ values[8,0])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[1,5] + values[1,6] + values[1,7] + values[1,8] + values[0,1]+ values[2,1]+ values[3,1]+ values[4,1]+ values[5,1]+ values[6,1]+ values[7,1]+ values[8,1])
    class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4] + values[2,5] + values[2,6] + values[2,7] + values[2,8] + values[0,2]+ values[1,2]+ values[3,2]+ values[4,2]+ values[5,2]+ values[6,2]+ values[7,2]+ values[8,2])
    class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[3,5] + values[3,6] + values[3,7] + values[3,8] + values[0,3]+ values[1,3]+ values[2,3]+ values[4,3]+ values[5,3] + values[6,3]+ values[7,3]+ values[8,3])
    
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(img_org, cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(img, cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(prediction_mask[0,:,:], cmap='jet')
    plt.show()    


print('MIOU: ' + str(np.mean(miou)))

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)