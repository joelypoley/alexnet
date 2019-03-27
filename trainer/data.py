from __future__ import division
from __future__ import print_function

import tensorflow as tf


def parse_fn_eval(example_proto):
    features = {
        "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
        "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    label = tf.one_hot(indices=parsed_features["image/class/label"], depth=1000)
    del parsed_features["image/class/label"]
    im = tf.image.decode_jpeg(parsed_features["image/encoded"], channels=3)
    im = tf.cast(im, tf.float32)
    im = tf.image.crop_to_bounding_box(
        image=im, offset_height=15, offset_width=15, target_height=227, target_width=227
    )

    return {"pixels": im}, label


def parse_fn_train(example_proto):
    features = {
        "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
        "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    label = tf.one_hot(indices=parsed_features["image/class/label"], depth=1000)
    del parsed_features["image/class/label"]
    im = tf.image.decode_jpeg(parsed_features["image/encoded"], channels=3)
    im = tf.cast(im, tf.float32)
    im = tf.image.random_crop(value=im, size=[227, 227, 3])
    im = tf.image.random_flip_left_right(image=im)

    return {"pixels": im}, label


def get_input_fn(file_pattern, eval, shuffle=False, batch_size=1, num_epochs=1):
    def input_fn():
        files = tf.data.Dataset.list_files(file_pattern)
        dataset = tf.data.TFRecordDataset(files)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=8)

        dataset = dataset.repeat(num_epochs)
        if eval:
            dataset = dataset.map(parse_fn_eval)
        else:
            dataset = dataset.map(parse_fn_train)

        dataset = dataset.batch(batch_size)
        return dataset

    return input_fn


# def json_serving_input_fn():
#     receiver_tensors = {"image/encoded": tf.placeholder(shape=[None], dtype=tf.float32)}
#     features = {
#         key: tensor for key, tensor in receiver_tensors.items()
#     }
#     return tf.estimator.export.ServingInputReceiver(
#         features=features, receiver_tensors=receiver_tensors
#     )
