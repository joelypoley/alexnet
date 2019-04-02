from __future__ import division
from __future__ import print_function

import tensorflow as tf

import task
import data


# I got pretty good (30% accuracy after 17 epochs) results when I removed the batch normalization and relu's and set the learning
# rate to 1e-5. The current configuration (which is what's given in the paper) doesn't learn much but I can't be bothered tinkering with it.
def model_fn(features, labels, mode, params):
    tf.summary.image("image", features["pixels"])
    layer = features["pixels"]
    # normalize images
    layer = layer - 128
    layer = tf.keras.layers.Conv2D(
        filters=96, kernel_size=11, strides=4, activation=tf.keras.activations.relu
    )(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(layer)

    layer = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=5,
        padding="same",
        bias_initializer="ones",
        activation=tf.keras.activations.relu,
    )(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(layer)

    layer = tf.keras.layers.Conv2D(
        filters=384, kernel_size=3, padding="same", activation=tf.keras.activations.relu
    )(layer)

    layer = tf.keras.layers.Conv2D(
        filters=384,
        kernel_size=3,
        padding="same",
        bias_initializer="ones",
        activation=tf.keras.activations.relu,
    )(layer)

    layer = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        padding="same",
        bias_initializer="ones",
        activation=tf.keras.activations.relu,
    )(layer)
    layer = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(layer)

    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(
        4096, bias_initializer="ones", activation=tf.keras.activations.relu
    )(layer)
    layer = tf.keras.layers.Dropout(0.5)(layer)
    layer = tf.keras.layers.Dense(
        4096, bias_initializer="ones", activation=tf.keras.activations.relu
    )(layer)
    logits = tf.keras.layers.Dense(1000)(layer)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"predictions": logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(
            labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1)
        )
        eval_metric_ops = {"accuracy": accuracy}
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = params["optimizer"]
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def create_estimator():
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            "optimizer": tf.contrib.opt.MomentumWOptimizer(
                weight_decay=0.0005,
                learning_rate=task.FLAGS.learning_rate,
                momentum=0.9,
            )
        },
        model_dir=task.FLAGS.job_dir,
    )
    return estimator
