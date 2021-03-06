"""
A minimal example of TBNNAM for binary classification without attention.
"""
from functools import partial
import numpy as np
import keras
import keras.backend as K
from keras import layers, optimizers, regularizers
from keras.initializers import Initializer
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed

from minimal_example_common_params import (
    num_samples,
    num_tokens_at_once,
    embedding_dim,
    l2_weight,
    num_entity_classes,
    num_event_classes,
    dim_ent,
    seed_value,
    MASK_VALUE_IGNORE_POSITION,
    beta,
    learning_rate,
    num_training_epochs,
    log_dir,
    lstm_dim,
    event_embedding_dim,
)

from test_labels import (
    biased_event_binary_crossentropy_np,
    make_random_labels,
    make_categorical_labels_including_negative_example_slot,
    make_categorical_labels_excluding_negative_example_slot,
    round_traditional_np
)

np.random.seed(seed_value)
set_random_seed(seed_value)
num_training_epochs = 100
num_event_classes = 47
max_event_labels_per_sample = 30
beta = 0.0


class ScaledRandomNormal(Initializer):
    def __init__(self, seed=None, scale_factor=1.0):
        self.mean = 0.0
        self.stddev = 1.0
        self.seed = seed
        self.scale_factor = scale_factor

    def __call__(self, shape, dtype=None):
        return self.scale_factor * K.random_normal(
            shape, self.mean, self.stddev, dtype=dtype, seed=self.seed
        )

    def get_config(self):
        return {
            "mean": self.mean,
            "stddev": self.stddev,
            "seed": self.seed,
            "scale_factor": self.scale_factor,
        }


def biased_mean_squared_error(
    y_true: np.ndarray, y_pred: np.ndarray, *, beta: float = 0.0
):
    return K.mean(K.square(y_pred - y_true) * K.square(1 + (y_true * beta)), axis=-1)


def event_binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray):
    return K.mean(K.mean(K.binary_crossentropy(y_true, y_pred), axis=0))


def biased_event_binary_crossentropy(
    y_true: np.ndarray, y_pred: np.ndarray, *, beta: float = 0.0
):
    return K.mean(
        K.mean(K.binary_crossentropy(y_true, y_pred) * (1 + (y_true * beta)), axis=0)
    )


K.clear_session()
word_embedding_input = layers.Input(
    shape=(num_tokens_at_once, embedding_dim), name="w_emb"
)
entity_type_input = layers.Input(shape=(num_tokens_at_once,), name="entity_type_input")
entity_type_embedding = layers.Embedding(
    num_entity_classes,
    dim_ent,
    name="ent_emb",
    embeddings_initializer=ScaledRandomNormal(seed=seed_value, scale_factor=0.01),
    embeddings_regularizer=regularizers.l2(l2_weight),
)(entity_type_input)
concat = layers.concatenate([word_embedding_input, entity_type_embedding])
mask = layers.Masking(mask_value=MASK_VALUE_IGNORE_POSITION)(concat)
encoder = layers.LSTM(
    lstm_dim,
    kernel_regularizer=regularizers.l2(l2_weight),
    bias_regularizer=regularizers.l2(l2_weight),
)(mask)
sigmoid = layers.Dense(
    num_event_classes,
    activation="sigmoid",
    kernel_regularizer=regularizers.l2(l2_weight),
    bias_regularizer=regularizers.l2(l2_weight),
    name="output",
)(encoder)
model = keras.Model(inputs=[word_embedding_input, entity_type_input], outputs=sigmoid)
loss_function = partial(biased_event_binary_crossentropy, beta=beta)
loss_function.__name__ = biased_event_binary_crossentropy.__name__
model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=loss_function)
model.summary()
custom_objects = {loss_function.__name__: loss_function}

embeddings = np.random.standard_normal(
    size=(num_samples, num_tokens_at_once, embedding_dim)
)
entity_types = np.random.random_integers(
    low=0, high=num_entity_classes - 1, size=(num_samples, num_tokens_at_once)
)
categorical_labels = make_random_labels(
    num_samples=num_samples,
    num_classes=num_event_classes,
    max_labels_per_sample=max_event_labels_per_sample,
)
labels = make_categorical_labels_excluding_negative_example_slot(
    categorical_labels, num_classes=num_event_classes
)
res = train_test_split(
    embeddings, entity_types, labels, test_size=0.10, shuffle=False, stratify=None
)
embeddings_train, embeddings_test = res[0], res[1]
entity_types_train, entity_types_test = res[2], res[3]
labels_train, labels_test = res[4], res[5]
training_batch_size = min(100, len(labels_test) // 2)
model.fit(
    [embeddings_train, entity_types_train],
    labels_train,
    epochs=num_training_epochs,
    batch_size=training_batch_size,
)

idx = 4
embeddings_single = embeddings_train[idx][np.newaxis, ...]
entity_types_single = entity_types_train[idx][np.newaxis, ...]
labels_single = labels_train[idx][np.newaxis, ...]
pred_single = model.predict([embeddings_single, entity_types_single])
print(biased_event_binary_crossentropy_np(labels_single, pred_single, beta=beta))
print(model.evaluate([embeddings_single, entity_types_single], labels_single))
pred_train = model.predict([embeddings_train, entity_types_train])
print(biased_event_binary_crossentropy_np(labels_train, pred_train, beta=beta))
print(model.evaluate([embeddings_train, entity_types_train], labels_train))
