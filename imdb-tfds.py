import tensorflow as tf
import numpy as np
import random
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_ds, test_ds = tfds.load('imdb_reviews', split=['train', 'test'], as_supervised=True)

train_data = []
train_labels = []

for example, label in train_ds:
    train_data.append(example.numpy().decode('utf-8'))
    train_labels.append(label.numpy())

test_data = []
test_labels = []

for example, label in test_ds:
    test_data.append(example.numpy().decode('utf-8'))
    test_labels.append(label.numpy())

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.97:
            print('Stopping training')
            self.model.stop_training = True

callback = MyCallback()

tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data)
word_index = tokenizer.word_index
total_words = len(word_index) + 1
train_sequences = tokenizer.texts_to_sequences(train_data)

embedding_dims = 16
max_sequence_len = 256

train_sequences = pad_sequences(train_sequences,
                                maxlen=max_sequence_len,
                                padding='post')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words,
                              embedding_dims,
                              input_length=max_sequence_len),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    ),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_sequences, train_labels,
                    epochs=10,
                    callbacks=[callback])

model.save('./saved_model')

correct_predictions = [0]
incorrect_predictions = [0]


def predict(model, data, labels):
    label = labels
    data = tokenizer.texts_to_sequences(data)
    data = pad_sequences(data,
                         maxlen=max_sequence_len,
                         padding='post')

    prediction = model.predict(data)

    prediction = prediction.argmax()

    if prediction > 0.5:
        prediction = 1
    else:
        prediction = 0

    if prediction - label == 0:
        print('Correct!')
        correct_predictions[0] += 1
    else:
        print('Incorrect!')
        incorrect_predictions[0] += 1


for _ in range(100):
    i = random.randint(0, 10000)
    predict(model=model, data=test_data[i], labels=test_labels[i])

test_accuracy = (correct_predictions[0] / (correct_predictions[0] + incorrect_predictions[0])) * 100

print('Test accuracy: ', test_accuracy, '%')
