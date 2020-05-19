"""
A minimal example of TBNNAM for binary classification without attention.
"""
from functools import partial
import numpy as np
import keras
import keras.backend as K
from keras import callbacks, layers, optimizers, regularizers
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


np.random.seed(seed_value)
set_random_seed(seed_value)


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


K.clear_session()
word_embedding_input = layers.Input(
    shape=(num_tokens_at_once, embedding_dim), name="w_emb"
)
entity_type_input = layers.Input(shape=(num_tokens_at_once,), name="entity_type_input")
event_type_input = layers.Input(shape=(1,), name="event_type_input")
entity_type_embedding = layers.Embedding(
    num_entity_classes,
    dim_ent,
    name="ent_emb",
    embeddings_initializer=ScaledRandomNormal(seed=seed_value, scale_factor=0.01),
    embeddings_regularizer=regularizers.l2(l2_weight),
)(entity_type_input)
event_type_embedding = layers.Embedding(
    num_event_classes,
    event_embedding_dim,
    name="evt_emb",
    embeddings_initializer=ScaledRandomNormal(seed=seed_value, scale_factor=0.01),
    embeddings_regularizer=regularizers.l2(l2_weight),
)(event_type_input)
concat = layers.concatenate([word_embedding_input, entity_type_embedding])
mask = layers.Masking(mask_value=MASK_VALUE_IGNORE_POSITION)(concat)
encoder = layers.LSTM(
    lstm_dim,
    kernel_regularizer=regularizers.l2(l2_weight),
    bias_regularizer=regularizers.l2(l2_weight),
)(mask)
mult = layers.Flatten()(
    layers.multiply(
        [
            layers.Reshape((-1, 1))(encoder),
            layers.Reshape((1, -1))(layers.Flatten()(event_type_embedding)),
        ]
    )
)
sigmoid = layers.Dense(
    1,
    activation="sigmoid",
    kernel_regularizer=regularizers.l2(l2_weight),
    bias_regularizer=regularizers.l2(l2_weight),
    name="output",
)(mult)
model = keras.Model(
    inputs=[word_embedding_input, entity_type_input, event_type_input], outputs=sigmoid
)
# TODO
model.layers[6].trainable = False
loss_function = partial(biased_mean_squared_error, beta=beta)
loss_function.__name__ = biased_mean_squared_error.__name__
model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=loss_function)
model.summary()
custom_objects = {"biased_mean_squared_error": loss_function}


embeddings = np.random.standard_normal(
    size=(num_samples, num_tokens_at_once, embedding_dim)
)
entity_types = np.random.random_integers(
    low=0, high=num_entity_classes - 1, size=(num_samples, num_tokens_at_once)
)
# one sentence at a time
event_types = np.random.random_integers(
    low=0, high=num_event_classes - 1, size=(num_samples, 1)
)
labels = np.random.random_integers(low=0, high=1, size=(num_samples, 1))
res = train_test_split(
    embeddings,
    entity_types,
    event_types,
    labels,
    test_size=0.10,
    shuffle=False,
    stratify=None,
)
embeddings_train, embeddings_test = res[0], res[1]
entity_types_train, entity_types_test = res[2], res[3]
event_types_train, event_types_test = res[4], res[5]
labels_train, labels_test = res[6], res[7]
training_batch_size = min(100, len(labels_test) // 2)
validation_data = (
    {
        "w_emb": embeddings_test,
        "entity_type_input": entity_types_test,
        "event_type_input": event_types_test,
    },
    {"output": labels_test},
)
model.fit(
    [embeddings_train, entity_types_train, event_types_train],
    labels_train,
    epochs=num_training_epochs,
    batch_size=training_batch_size,
    validation_data=validation_data,
    callbacks=[
        callbacks.TensorBoard(
            log_dir=str(log_dir / "m3"),
            histogram_freq=1,
            write_graph=True,
            write_grads=True,
            update_freq="batch",
        )
    ],
)


def _compute_elemwise_op_output_shape(shape1, shape2):
    """Computes the shape of the resultant of an elementwise operation.

    # Arguments
        shape1: tuple or None. Shape of the first tensor
        shape2: tuple or None. Shape of the second tensor

    # Returns
        expected output shape when an element-wise operation is
        carried out on 2 tensors with shapes shape1 and shape2.
        tuple or None.

    # Raises
        ValueError: if shape1 and shape2 are not compatible for
            element-wise operations.
    """
    if None in [shape1, shape2]:
        return None
    elif len(shape1) < len(shape2):
        return _compute_elemwise_op_output_shape(shape2, shape1)
    elif not shape2:
        return shape1
    output_shape = list(shape1[: -len(shape2)])
    for i, j in zip(shape1[-len(shape2) :], shape2):
        if i is None or j is None:
            output_shape.append(None)
        elif i == 1:
            output_shape.append(j)
        elif j == 1:
            output_shape.append(i)
        else:
            if i != j:
                raise ValueError(
                    "Operands could not be broadcast "
                    "together with shapes " + str(shape1) + " " + str(shape2)
                )
            output_shape.append(i)
    return tuple(output_shape)
