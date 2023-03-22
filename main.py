import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import os

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.data import AUTOTUNE
from tensorflow.keras.layers.experimental import preprocessing

# Change the directory
os.chdir("<your-work-directory>/train/Images")      # "train" is the directory of the combined dataset of the two training set


## Split the data
def Split(directory, validation = 0.3, inception = False):  
    if inception == False:
        IMG_DIM = 224      # for ResNet v2
    else:
        IMG_DIM = 299       # for Inception ResNet v2
    
    BATCH_SIZE = 32

    train_ds = image_dataset_from_directory(directory,
                                                image_size=(IMG_DIM,IMG_DIM),
                                                label_mode="categorical",
                                                batch_size=BATCH_SIZE,
                                                validation_split = validation,
                                                subset="training",
                                                seed=117,
                                                shuffle=True)

    val_ds = image_dataset_from_directory(directory,
                                                image_size=(IMG_DIM,IMG_DIM),
                                                label_mode="categorical",
                                                batch_size=BATCH_SIZE,
                                                validation_split = validation,
                                                subset="validation",
                                                seed=117,
                                                shuffle=True)
    
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


## Data Augmentation
def dataAug(train_ds):
    aug1 = preprocessing.RandomFlip("horizontal")
    ds_1 = train_ds.map(lambda x, y: (aug1(x), y), num_parallel_calls=AUTOTUNE)
    aug2 = preprocessing.RandomRotation(0.2)
    ds_2 = train_ds.map(lambda x, y: (aug2(x), y), num_parallel_calls=AUTOTUNE)
    aug3 = layers.RandomTranslation(0.3, 0.3)
    ds_3 = train_ds.map(lambda x, y: (aug3(x), y), num_parallel_calls=AUTOTUNE)
    aug4 = layers.RandomBrightness(0.4)
    ds_4 = train_ds.map(lambda x, y: (aug4(x), y), num_parallel_calls=AUTOTUNE)
    aug5 = layers.RandomContrast(0.4)
    ds_5 = train_ds.map(lambda x, y: (aug5(x), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.concatenate(ds_1).concatenate(ds_2).concatenate(ds_3).concatenate(ds_4).concatenate(ds_5)
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    return train_ds


## Run the model
def RUN(train_ds, val_ds, category, inception = False, trainable = False, epoch = 60, earlystop = 15):
    '''Default model selection is the ResNet v2 50 without fine-tuning, while Inception ResNet v2 and Fine-tune are optional'''
    ## Build the modle
    n_classes = train_ds.element_spec[1].shape[-1]

    if inception == False:
        IMG_DIM = 224      # for ResNet v2
        basemodel_path = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5"
        #basemodel_path = "<directory for storing the downloaded model>/imagenet_resnet_v2_50_classification_5"
    else:
        IMG_DIM = 299       # for Inception ResNet v2
        basemodel_path = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5"
        #basemodel_path = "<directory for storing the downloaded model>/imagenet_inception_resnet_v2_classification_5"
   

    model = Sequential([
        hub.KerasLayer(basemodel_path,
        trainable = trainable,
        arguments = dict(batch_norm_momentum = 0.997)),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(n_classes, activation='softmax')
        ])
    model.build([None, IMG_DIM, IMG_DIM, 3])
    model.summary()

    ## Complie the model
    model.compile(optimizer='adam',
              loss="categorical_crossentropy",     # SparseCategoricalCrossentropy used for interger Yi; CategoricalCrossentropy used for one-hot Yi
              metrics=['accuracy'])
    
    ## Fit the model
    epochs = epoch

    # Store the weights of the model which has the best performance on the validation set
    checkpoint = ModelCheckpoint(f"weights_{category}.hdf5", monitor='val_accuracy', mode="max", verbose = 1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=earlystop)
    callback_list = [checkpoint, early_stopping]

    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callback_list
    )

    return history


## Visualization
def Vis(history, category):
    ## Visualize the output
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{category}: Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{category}: Training and Validation Loss')
    plt.savefig(f'{category}.png')
    plt.show()


## Examples of using this py files 
# Lapel_Design
lapel_train, lapel_val = Split("lapel_design_labels")
#lapel_train = dataAug(lapel_train)
hist_lapel = RUN(lapel_train, lapel_val, "Lapel_Design_ResNetV2", False, False, 10)
Vis(hist_lapel, "Lapel_Design")
hist_lapel_tr = RUN(lapel_train, lapel_val, "Lapel_Design_ResNetV2_Tr", False, True, 20)
Vis(hist_lapel_tr, 'Lapel_Design_tr')
hist_lapel_incp = RUN(lapel_train, lapel_val, "Lapel_Design_IncpResNetV2", True, False, 10)
Vis(hist_lapel_incp, 'Lapel_Design_incp')
hist_lapel_incp_tr = RUN(lapel_train, lapel_val, "Lapel_Design_ResNetV2_Tr", True, True, 20)
Vis(hist_lapel_incp_tr, 'Lapel_Design_incp_tr')
