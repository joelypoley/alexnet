from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

import data
import model


def initialise_flags(args_parser):
    args_parser.add_argument("--train", required=True)
    args_parser.add_argument("--max-steps", type=int, default=None)
    args_parser.add_argument("--eval-steps", type=int, default=None)
    args_parser.add_argument("--num-epochs", type=int, default=90)
    args_parser.add_argument("--batch-size", type=int, default=1)
    args_parser.add_argument("--eval", required=True)
    args_parser.add_argument("--learning-rate", type=float, default=1e-2)
    args_parser.add_argument("--throttle-secs", type=int, default=600)
    args_parser.add_argument("--job-dir", required=True)
    args_parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
        default="INFO",
    )

    return args_parser.parse_args()


def main():
    tf.logging.set_verbosity(FLAGS.verbosity)

    train_spec = tf.estimator.TrainSpec(
        input_fn=data.get_input_fn(
            file_pattern=FLAGS.train,
            eval=False,
            shuffle=False,
            batch_size=FLAGS.batch_size,
            num_epochs=FLAGS.num_epochs,
        ),
        max_steps=FLAGS.max_steps,
    )
    # exporter = tf.estimator.FinalExporter(
    #     "estimator",
    #     data.json_serving_input_fn,
    #     as_text=False,  # change to true if you want to export the model as readable text
    # )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=data.get_input_fn(file_pattern=FLAGS.eval, eval=True, batch_size=1024),
        steps=FLAGS.eval_steps,
        throttle_secs=FLAGS.throttle_secs,
        # exporters=[exporter],
    )

    estimator = model.create_estimator()
    tf.logging.log(
        tf.logging.INFO,
        "About to start training and evaulating. To see results type\ntensorboard --logdir={}\nin another window.".format(
            FLAGS.job_dir
        ),
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    tf.logging.log(
        tf.logging.INFO,
        "Finished training and evaulating. To see results type\ntensorboard --logdir={}\nin another window.".format(
            FLAGS.job_dir
        ),
    )


args_parser = argparse.ArgumentParser()
FLAGS = initialise_flags(args_parser)

if __name__ == "__main__":
    main()
