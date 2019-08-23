from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

FIT_BATCH_SIZE = 64
NUM_EPOCHS = 3

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

# Generate dummy testing data
x_test = np.random.random((1000, timesteps, data_dim))
y_test = np.random.random((1000, num_classes))

# expected input data shape: (batch_size, timesteps, data_dim)
model1 = Sequential()
model1.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model1.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model1.add(LSTM(32))  # return a single vector of dimension 32
model1.add(Dense(10, activation='softmax'))

model2 = Sequential()
model2.add(LSTM(32, return_sequences=True,
                input_shape=(timesteps, data_dim)))
model2.add(LSTM(32, return_sequences=True))
model2.add(Bidirectional(LSTM(32)))
model2.add(Dense(10, activation='softmax'))

model3 = Sequential()
model3.add(LSTM(32, return_sequences=False,
                input_shape=(timesteps, data_dim)))
model3.add(Dense(10, activation='softmax'))

for model in (model1, model2, model3):
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=FIT_BATCH_SIZE, epochs=NUM_EPOCHS,
              validation_data=(x_val, y_val))
    res = model.evaluate(x_test, y_test)
    print(res)
