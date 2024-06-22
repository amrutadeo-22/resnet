import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pathlib


dataset = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
directory = tf.keras.utils.get_file('flower_photos', origin=dataset, untar=True)
data_directory = pathlib.Path(directory)


img_height, img_width = 180, 180
batch_size = 32


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


class_names = train_ds.class_names


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(6):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


def identity_block(x, filters):
    f1, f2 = filters

    x_shortcut = x


    x = layers.Conv2D(filters=f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)


    x = layers.Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)


    x = layers.Add()([x, x_shortcut])
    x = layers.ReLU()(x)

    return x


def convolutional_block(x, filters, strides=(2, 2)):
    f1, f2 = filters

    x_shortcut = layers.Conv2D(filters=f2, kernel_size=(1, 1), strides=strides, padding='valid')(x)
    x_shortcut = layers.BatchNormalization()(x_shortcut)


    x = layers.Conv2D(filters=f1, kernel_size=(3, 3), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)


    x = layers.Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)


    x = layers.Add()([x, x_shortcut])
    x = layers.ReLU()(x)

    return x


def ResNet18(input_shape, classes):
    inputs = keras.Input(shape=input_shape)


    x = layers.ZeroPadding2D((3, 3))(inputs)


    x = layers.Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)


    x = convolutional_block(x, filters=[64, 64], strides=(1, 1))
    x = identity_block(x, filters=[64, 64])


    x = convolutional_block(x, filters=[128, 128])
    x = identity_block(x, filters=[128, 128])


    x = convolutional_block(x, filters=[256, 256])
    x = identity_block(x, filters=[256, 256])


    x = convolutional_block(x, filters=[512, 512])
    x = identity_block(x, filters=[512, 512])


    x = layers.GlobalAveragePooling2D()(x)


    x = layers.Dense(classes, activation='softmax')(x)


    model = keras.Model(inputs=inputs, outputs=x, name='ResNet18')

    return model


input_shape = (img_height, img_width, 3)
num_classes = len(class_names)


resnet18_model = ResNet18(input_shape, num_classes)


resnet18_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = resnet18_model.fit(train_ds, validation_data=validation_ds, epochs=10)

val_loss, val_accuracy = resnet18_model.evaluate(validation_ds)

print(f"Validation Loss: {val_loss:.2f}")
print(f"Validation Accuracy: {val_accuracy:.2f}")