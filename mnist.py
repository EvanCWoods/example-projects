import tensorflow as tf
import numpy as np
import random
import keras
from keras.preprocessing.image import ImageDataGenerator

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

train_data = train_data / 255
test_data = test_data / 255

train_data = tf.expand_dims(train_data, axis=-1)
test_data = tf.expand_dims(test_data, axis=-1)

train_datagen = ImageDataGenerator(height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   shear_range=0.2,
                                   rotation_range=20)
test_datagen = ImageDataGenerator(height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  shear_range=0.2,
                                  rotation_range=20)

train_datagen.fit(train_data)
test_datagen.fit(test_data)


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.97:
            print('Stopping training.')
            self.model.stop_training = True

callback = MyCallback()


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=[28,28,1]),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    epochs=10,
                    callbacks=[callback])


correct = [0]
incorrect = [0]


def predict(data, model, label):
    data = np.array(data)
    data = data.reshape(1, 28, 28, 1)

    prediction = model.predict(data)
    prediction = prediction.argmax()

    print('Prediction:', prediction)
    print('Label: ', label)

    if prediction.argmax() - label == 0:
        print('Correct')
        correct[0] += 1
    else:
        print('Incorrect')
        incorrect[0] += 1


for _ in range(100):
    i = random.randint(0, 10000)
    predict(data=test_data[i], model=model, label=test_labels[i])