import numpy as np
from tensorflow.keras import layers, Sequential

np.set_printoptions(linewidth=200)

vocab_size = 13
embedding_dim = 12
input_length = 10

# embedding = layers.Embedding(vocab_size, embedding_dim, input_length=input_length)
# Don't fix the input dimension
embedding = layers.Embedding(vocab_size, embedding_dim, input_length=None)
model = Sequential([embedding])
model.compile("rmsprop", "mse")
model.summary()
# The model will take as input an integer matrix of size (batch, input_length), and the largest
# integer (i.e. word index) in the input should be no larger than `vocab_size - 1` (vocabulary
# size).  Now model.output_shape is (None, `input_length`, `embedding_dim`), where `None` is the
# batch dimension.
batch_size = 4
input_array = np.random.randint(vocab_size, size=(batch_size, input_length))
output_array = model.predict(input_array)
# assert output_array.shape == (batch_size, input_length, embedding_dim)
print(input_array)
print(output_array)

# input_array_2 = np.asarray([[0], [4, 3], [1, 1, 1, 1], [], [1, 2]])
# output_array_2 = model.predict(input_array_2)
# print(input_array_2)
# print(output_array_2)

input_array_2 = np.asarray([
    [0, 0, 0, 0],
    [4, 3, 0, 0],
    [1, 2, 10, 11],
    [1, 0, 0, 0],
    [1, 2, 0, 0]
])
output_array_2 = model.predict(input_array_2)
print(input_array_2)
print(output_array_2)

embedding_2 = layers.Embedding(
    vocab_size, embedding_dim, input_length=None, mask_zero=True
)
model_2 = Sequential([embedding_2])
model_2.compile("rmsprop", "mse")
model_2.summary()
output_array_3 = model_2.predict(input_array_2)
print(output_array_3)
