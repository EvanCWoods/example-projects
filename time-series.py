import tensorflow as tf
import numpy as np
import random
import csv
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

time_steps = []
sunspots = []

with open('Sunspots.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        sunspots.append(float(row[2]))
        time_steps.append(int(row[0]))


series = np.array(sunspots)
time = np.array(time_steps)


split_time = 3000
train_data = series[:split_time]
train_time = time[:split_time]
test_data = series[split_time:]
test_time = time[split_time:]

train_features = tf.expand_dims(train_data, axis=-1)
train_targets = tf.expand_dims(train_data, axis=-1)
test_features = tf.expand_dims(test_data, axis=-1)
test_targets = tf.expand_dims(test_data, axis=-1)


length = 11
batch_size = 32

train_gen = TimeseriesGenerator(data=train_features,
                                targets=train_targets,
                                length=length,
                                sampling_rate=1,
                                batch_size=batch_size)

model = tf.keras.Sequential([
  tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='causal', activation='relu', input_shape=[None, 1]),
  tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM(64, return_sequences=True)
  ),
  tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM(64, return_sequences=True)
  ),
  tf.keras.layers.LSTM(64),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1)
])

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mse'])

history = model.fit(train_gen, epochs=25)


def predict(model, features, targets):
    predictions = model.predict(TimeseriesGenerator(data=features,
                                                    targets=targets,
                                                    length=length,
                                                    sampling_rate=1,
                                                    batch_size=batch_size))

    for i in range(0, 100):
        target = targets[i]
        prediciton = predictions[i]

        loss = prediciton - target

        print()
        print('Incrament: ', i)
        print('Loss: ', np.array(loss[0]))
        print('prediction: ', prediciton)
        print('Target: ', np.array(target[0]))
        print()

predict(model=model, features=test_features, targets=test_targets)