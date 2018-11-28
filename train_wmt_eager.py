# coding: utf-8
"""
reference:
  https://github.com/tensorflow/models/blob/master/official/mnist/mnist_eager.py
  trainer_lib.create_experiment
"""

import tensorflow as tf
import os
from tensor2tensor import problems
from tensor2tensor import models
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import registry

import time

# Enable TF Eager execution
tfe = tf.contrib.eager
tfe.enable_eager_execution()

# Other setup
tf.logging.set_verbosity(tf.logging.DEBUG)

# Setup some directories
problem = "translate_enzh_wmt8k"
model = "transformer"
hparams_set = "transformer_test"
checkpoint_dir = 't2t_train/' + problem + "/" + model + "-" + hparams_set
data_dir = 't2t_data/' + problem

# Fetch the problem
enzh_problem = problems.problem(problem)

hparams = trainer_lib.create_hparams(hparams_set, data_dir=data_dir, problem_name=problem)
run_config = trainer_lib.create_run_config(hparams, model_dir=checkpoint_dir)

trainer_lib.add_problem_hparams(hparams, problem)

ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)


def train(model, optimizer, dataset, step_counter, log_interval=None):
  """Trains model on `dataset` using `optimizer`."""
  start = time.time()
  for (batch, (features, _)) in enumerate(dataset):  # what is _？run_config?
    with tf.contrib.summary.record_summaries_every_n_global_steps(
            10, global_step=step_counter):
      with tf.GradientTape() as tape:
        logits, losses_dict = model(features)
        # Summarize losses
        model._summarize_losses(losses_dict)
        # Accumulate losses
        loss = sum(losses_dict[key] for key in sorted(losses_dict.keys()))

      print("model.variables", model.variables)  # why is empty?
      grads = tape.gradient(loss, model.variables)
      optimizer.apply_gradients(
        zip(grads, model.variables), global_step=step_counter)
      if log_interval and batch % log_interval == 0:
        rate = log_interval / (time.time() - start)
      print('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss, rate))
      start = time.time()
    return 0


def test(model, dataset):
  """Perform an evaluation of `model` on the examples from `dataset`."""
  pass
  # avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
  # accuracy = tfe.metrics.Accuracy('accuracy', dtype=tf.float32)
  #
  # for (images, labels) in dataset:
  #   logits = model(images, training=False)
  #   avg_loss(loss(logits, labels))
  #   accuracy(
  #       tf.argmax(logits, axis=1, output_type=tf.int64),
  #       tf.cast(labels, tf.int64))
  # print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
  #       (avg_loss.result(), 100 * accuracy.result()))
  # with tf.contrib.summary.always_record_summaries():
  #   tf.contrib.summary.scalar('loss', avg_loss.result())
  #   tf.contrib.summary.scalar('accuracy', accuracy.result())

def run_eager():
  """Run training and eval loop in eager mode."""
  # Load the datasets
  dataset_train = enzh_problem.input_fn(tf.estimator.ModeKeys.TRAIN, hparams)
  dataset_eval = enzh_problem.input_fn(tf.estimator.ModeKeys.EVAL, hparams)

  # Create the model and optimizer
  translate_model = registry.model(model)(hparams, tf.estimator.ModeKeys.TRAIN)  # continues train and eval呢？
  optimizer = tf.train.MomentumOptimizer(0.01, 0.5)
  train(translate_model, optimizer, dataset_train, 5)

  # Create file writers for writing TensorBoard summaries.
  output_dir = False
  if output_dir:
    # Create directories to which summaries will be written
    # tensorboard --logdir=<output_dir>
    # can then be used to see the recorded summaries.
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)
  else:
    train_dir = None
    test_dir = None

  summary_writer = tf.contrib.summary.create_file_writer(
    train_dir, flush_millis=10000)
  test_summary_writer = tf.contrib.summary.create_file_writer(
    test_dir, flush_millis=10000, name='test')

  # Create and restore checkpoint (if one exists on the path)
  step_counter = tf.train.get_or_create_global_step()
  checkpoint = tf.train.Checkpoint(
    model=model, optimizer=optimizer, step_counter=step_counter)
  # Restore variables on creation if a checkpoint exists.
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  # Train and evaluate for a set number of epochs.
  train_epochs = 10
  log_interval = 2
  device = "cpu:0"
  with tf.device(device):
    for _ in range(train_epochs):
      start = time.time()
      with summary_writer.as_default():
        train(model, optimizer, dataset_train, step_counter,
              log_interval)
      end = time.time()
      print('\nTrain time for epoch #%d (%d total steps): %f' %
            (checkpoint.save_counter.numpy() + 1,
             step_counter.numpy(),
             end - start))
      with test_summary_writer.as_default():
        test(model, dataset_eval)
      checkpoint.save(checkpoint_dir)



if __name__ == "__main__":
  run_eager()

