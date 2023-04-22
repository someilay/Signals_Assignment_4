import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import tensorflow as tf
from tensorflow import keras


if __name__ == '__main__':
    IMAGE_SHAPE = (369, 496)
    VALID_DATA_DIR = 'corrupted_spec/'  # replace with your generated test set in here

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    )

    valid_generator = datagen.flow_from_directory(
        VALID_DATA_DIR,
        shuffle=False,
        target_size=IMAGE_SHAPE,
    )

    model = keras.models.load_model('my_model')
    model.evaluate(valid_generator)
