import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train(model):

    # Check the number of images in the dataset
    train = []
    label = []

    # os.listdir returns the list of files in the folder, in this case image class names
    for i in os.listdir('./train'):
        train_class = os.listdir(os.path.join("train", i))
        for j in train_class:
            img = os.path.join('train', i, j)
            train.append(img)
            label.append(i)

    # print(f"Number of train images {len(train)}")

    # check the number of images in each class in the training dataset

    No_images_per_class = []
    Class_name = []
    for i in os.listdir('./train'):
        train_class = os.listdir(os.path.join('train', i))
        No_images_per_class.append(len(train_class))
        Class_name.append(i)
        print('Number of images in {} = {} \n'.format(i, len(train_class)))

    retina_df = pd.DataFrame({'Image': train, 'Labels': label})
    # retina_df
    retina_df = retina_df.groupby('Labels').apply(lambda x: x.sample(
        n=min(300, x.shape[0]))).reset_index(drop=True)

    # Shuffle the data and split it into training and testing
    retina_df = shuffle(retina_df)
    train, test = train_test_split(retina_df, test_size=0.2)
    # Create run-time augmentation on training and test dataset
    # For training datagenerator, we add normalization, shear angle, zooming range and horizontal flip
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1./255,
        validation_split=0.15)

    # For test datagenerator, we only normalize the data.
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Creating datagenerator for training, validation and test dataset.

    train_generator = train_datagen.flow_from_dataframe(
        train,
        directory='./',
        x_col="Image",
        y_col="Labels",
        target_size=(256, 256),
        color_mode="rgb",
        class_mode="categorical",
        #     validate_filenames=False,
        batch_size=16,
        subset='training')

    validation_generator = train_datagen.flow_from_dataframe(
        train,
        directory='./',
        x_col="Image",
        y_col="Labels",
        target_size=(256, 256),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=16,
        #     validate_filenames=False,
        # batch_size=32,
        subset='validation')

    test_generator = test_datagen.flow_from_dataframe(
        test,
        directory='./',
        x_col="Image",
        y_col="Labels",
        target_size=(256, 256),
        validate_filenames=False,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=16,
        # batch_size=32
    )

    # using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
    earlystopping = EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=15)

    # save the best model with lower validation loss
    checkpointer = ModelCheckpoint(
        filepath="weights_14m.hdf5", verbose=1, save_best_only=True)

    history = model.fit(train_generator, steps_per_epoch=train_generator.n // 16, epochs=10, validation_data=validation_generator,
                        validation_steps=validation_generator.n // 16, callbacks=[checkpointer, earlystopping])
    model.save("test.hdf5")
