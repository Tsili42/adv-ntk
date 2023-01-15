# Code adjusted from 
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Robust training.
================

The following code trains a convolutional neural network (CNN) to be robust
with respect to the projected gradient descent (PGD) method.

The Projected Gradient Descent Method (PGD) is a simple yet effective method to
generate adversarial images. At each iteration, it adds a small perturbation
in the direction of the sign of the gradient with respect to the input followed
by a projection onto the infinity ball. The gradient sign ensures this
perturbation locally maximizes the objective, while the projection ensures this
perturbation stays on the boundary of the infinity ball.

References
----------
  Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining
  and harnessing adversarial examples." https://arxiv.org/abs/1412.6572

  Madry, Aleksander, et al. "Towards deep learning models resistant to
  adversarial attacks." arXiv preprint arXiv:1706.06083 (2017).
"""

import datetime
import pickle

from absl import app
from absl import flags
from flax import linen as nn
import jax
from jax import numpy as jnp
from jaxopt import loss
from jaxopt import OptaxSolver
from jaxopt import tree_util
from matplotlib import pyplot as plt
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import neural_tangents as nt
import os

dataset_names = [
    "mnist", "kmnist", "emnist", "fashion_mnist", "cifar10", "cifar100"
]

flags.DEFINE_float("l2reg", 0, "L2 regularization.")
flags.DEFINE_float("learning_rate", 1e-2, "Learning rate.")
flags.DEFINE_float(
    "epsilon", 8/255,
    "Adversarial perturbations will be constrained to lie within the L-infinity ball of radius epsilon."
)
flags.DEFINE_float(
    "alpha", 2/255,
    "Step of PGD."
)
flags.DEFINE_integer("pgd_steps", 20, "Number of pgd steps for solving inner max loop.")
flags.DEFINE_enum("dataset", "cifar10", dataset_names, "Dataset to train on.")
flags.DEFINE_integer("kernel_batch", 10, "Batch size used for kernel estimation.")
flags.DEFINE_integer("epochs", 200, "Number of passes over the dataset.")
flags.DEFINE_integer("lin_epoch", 0, "Epoch to linearize.")
flags.DEFINE_integer("train_batch_size", 250, "Batch size at train time.")
flags.DEFINE_integer("test_batch_size", 256, "Batch size at test time.")
flags.DEFINE_bool("adv_train", True, "Whether to adversarially train or not")
flags.DEFINE_bool("binary", False, "Whether the task is binary or not - 0 vs 1, as a convention")

FLAGS = flags.FLAGS

def load_dataset(split, *, is_training, batch_size):
  """Load dataset using tensorflow_datasets."""
  version = 3
  ds, ds_info = tfds.load(
      f"{FLAGS.dataset}:{version}.*.*",
      as_supervised=True,  # remove useless keys
      split=split,
      with_info=True)
  ds = ds.cache().repeat()
  if FLAGS.binary:
    ds = ds.filter(lambda x, y: y == 0 or y == 1)
    ds = ds.map(lambda x, y: (x, 2*y - 1))
  # if is_training:
  #   ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds)), ds_info


class CNN(nn.Module):
  """A simple CNN model."""
  num_classes: int

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_classes)(x)
    return x

class CNN_mega(nn.Module):
  """A simple CNN model."""
  num_classes: int

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=self.num_classes)(x)
    return x

class FC(nn.Module):
  """A simple FC model."""
  num_classes: int

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=10000)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10000)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_classes)(x)
    return x


def main(argv):
  del argv

  LINEARIZED = False # whether we want to perform linearized training after a few epochs
  # print(FLAGS.epsilon, FLAGS.pgd_steps, FLAGS.lin_epoch, FLAGS.adv_train, 'with sanity check')

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")
  train_ds, ds_info = load_dataset("train", is_training=True,
                                   batch_size=FLAGS.train_batch_size)
  test_ds, _ = load_dataset("test", is_training=False,
                            batch_size=FLAGS.test_batch_size)
  input_shape = (1,) + ds_info.features["image"].shape
  if FLAGS.binary:
    num_classes = 1
    iter_per_epoch_train = 10000 // FLAGS.train_batch_size
    iter_per_epoch_test = 2000 // FLAGS.test_batch_size
  else:
    num_classes = ds_info.features["label"].num_classes
    iter_per_epoch_train = ds_info.splits['train'].num_examples // FLAGS.train_batch_size
    iter_per_epoch_test = ds_info.splits['test'].num_examples // FLAGS.test_batch_size


  net = CNN_mega(num_classes)
  f = lambda params, inputs: net.apply({"params": params}, inputs) # suitable for ntk calculations
  # kernel_fn = nt.batch(nt.empirical_ntk_fn(f, trace_axes=(), vmap_axes=0), device_count=-1, batch_size=20) #used for mnist
  kernel_fn = nt.batch(nt.empirical_ntk_fn(f, trace_axes=(), vmap_axes=0), device_count=-1, batch_size=FLAGS.kernel_batch) #used for cifar10

  @jax.jit
  def accuracy(params, data):
    inputs, labels = data
    x = inputs.astype(jnp.float32)
    # logits = jax.lax.cond(LINEARIZED, f_lin, f, params, x)
    logits = net.apply({"params": params}, x)
    if FLAGS.binary:
      return jnp.mean(jnp.sign(logits.flatten()) == labels)
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)

  logistic_loss = jax.vmap(loss.multiclass_logistic_loss)
  if (FLAGS.binary):
      logistic_loss = jax.vmap(loss.binary_logistic_loss)


  @jax.jit
  def lin_accuracy(params, data):
    inputs, labels = data
    x = inputs.astype(jnp.float32)
    logits = f_lin(params, x)
    if FLAGS.binary:
      return jnp.mean(jnp.sign(logits.flatten()) == labels)
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


  @jax.jit
  def loss_fun(params, l2reg, data):
    """Compute the loss of the network."""
    inputs, labels = data
    x = inputs.astype(jnp.float32)
    # logits = jax.lax.cond(LINEARIZED, f_lin, f, params, x)
    logits = net.apply({"params": params}, x)
    sqnorm = tree_util.tree_l2_norm(params, squared=True)
    if FLAGS.binary:
      loss_value = jnp.mean(logistic_loss((labels+1)/2, logits))
    else:
      loss_value = jnp.mean(logistic_loss(labels, logits))
    return loss_value + 0.5 * l2reg * sqnorm

  @jax.jit
  def lin_loss_fun(params, l2reg, data):
      """Compute the loss of the network."""
      inputs, labels = data
      x = inputs.astype(jnp.float32)
      # logits = jax.lax.cond(LINEARIZED, f_lin, f, params, x)
      logits = f_lin(params, x)
      # logits = net.apply({"params": params}, x)
      sqnorm = tree_util.tree_l2_norm(params, squared=True)
      if FLAGS.binary:
          loss_value = jnp.mean(logistic_loss((labels+1)/2, logits))
      else:
          loss_value = jnp.mean(logistic_loss(labels, logits))
      return loss_value + 0.5 * l2reg * sqnorm

  @jax.jit
  def pgd_attack(image, label, params, epsilon=0.3, maxiter=20):
    """PGD attack on the L-infinity ball with radius epsilon.

    Args:
      image: array-like, input data for the CNN
      label: integer, class label corresponding to image
      params: tree, parameters of the model to attack
      epsilon: float, radius of the L-infinity ball.
      maxiter: int, number of iterations of this algorithm.

    Returns:
      perturbed_image: Adversarial image on the boundary of the L-infinity ball
        of radius epsilon and centered at image.

    Notes:
      PGD attack is described in (Madry et al. 2017),
      https://arxiv.org/pdf/1706.06083.pdf
    """
    image_perturbation = jnp.zeros_like(image)
    def adversarial_loss(perturbation):
      return loss_fun(params, 0, (image + perturbation, label))

    grad_adversarial = jax.grad(adversarial_loss)
    if (maxiter == 1):
        step = epsilon
    else:
        step = FLAGS.alpha
    for _ in range(maxiter):
      # compute gradient of the loss wrt to the image
      sign_grad = jnp.sign(grad_adversarial(image_perturbation))

      image_perturbation += step * sign_grad
      # projection step onto the L-infinity ball centered at image
      image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

    # clip the image to ensure pixels are between 0 and 1
    return jnp.clip(image + image_perturbation, 0, 1)


  @jax.jit
  def lin_pgd_attack(image, label, params, epsilon=0.3, maxiter=20):
      """PGD attack on the L-infinity ball with radius epsilon.

      Args:
        image: array-like, input data for the CNN
        label: integer, class label corresponding to image
        params: tree, parameters of the model to attack
        epsilon: float, radius of the L-infinity ball.
        maxiter: int, number of iterations of this algorithm.

      Returns:
        perturbed_image: Adversarial image on the boundary of the L-infinity ball
          of radius epsilon and centered at image.

      Notes:
        PGD attack is described in (Madry et al. 2017),
        https://arxiv.org/pdf/1706.06083.pdf
      """
      image_perturbation = jnp.zeros_like(image)
      def adversarial_loss(perturbation):
        return lin_loss_fun(params, 0, (image + perturbation, label))

      grad_adversarial = jax.grad(adversarial_loss)
      if (maxiter == 1):
          step = epsilon
      else:
          step = FLAGS.alpha
      for _ in range(maxiter):
        # compute gradient of the loss wrt to the image
        sign_grad = jnp.sign(grad_adversarial(image_perturbation))

        # heuristic step-size 2 eps / maxiter
        image_perturbation += step * sign_grad
        # projection step onto the L-infinity ball centered at image
        image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

      # clip the image to ensure pixels are between 0 and 1
      return jnp.clip(image + image_perturbation, 0, 1)

  # Initialize solver and parameters.
  solver = OptaxSolver(
      opt=optax.sgd(FLAGS.learning_rate),
      fun=loss_fun,
      maxiter=FLAGS.epochs * iter_per_epoch_train)
  key = jax.random.PRNGKey(0)
  params = net.init(key, jnp.zeros(input_shape))["params"]

  state = solver.init_state(params)
  start = datetime.datetime.now().replace(microsecond=0)
  jitted_update = jax.jit(solver.update)


  accuracy_train = []
  accuracy_test = []
  adversarial_accuracy_train = []
  adversarial_accuracy_test = []
  type_of_train = {
                    True: 'adv_train' + str(FLAGS.pgd_steps),
                    False: 'clean_train'
                    }
  type_of_task = {
                    True: 'binary',
                    False: ''
                    }
  id_train = type_of_train[FLAGS.adv_train]
  id_task = type_of_task[FLAGS.binary]
  path = f'./sgd_robust_train_results/linearize/linrz_on{FLAGS.lin_epoch}/{FLAGS.dataset}{id_task}/{id_train}'
  os.makedirs(path, exist_ok=True)
  os.makedirs(path+'/samples', exist_ok=True)
  os.makedirs(path+'/kernels', exist_ok=True)
  os.makedirs(path+'/results', exist_ok=True)
  count = 0
  for it in range(solver.maxiter):

    if (it == iter_per_epoch_train*FLAGS.lin_epoch):
        LINEARIZED = True
        f_lin = nt.linearize(f, params)

        kernel_fn = nt.batch(nt.empirical_ntk_fn(f_lin, trace_axes=(), vmap_axes=0), device_count=-1, batch_size=FLAGS.kernel_batch) # used for cifar10

        # Update solver
        solver = OptaxSolver(
            opt=optax.sgd(FLAGS.learning_rate),
            fun=lin_loss_fun,
            maxiter=FLAGS.epochs * iter_per_epoch_train)

        state = solver.init_state(params)
        jitted_update = jax.jit(solver.update)

    # training loop
    images, labels = next(train_ds)
    images = images.astype(jnp.float32) / 255
    if (it == 0):
        k_ss = kernel_fn(images, None, params)
        jnp.save(path+f'/kernels/kernel_epoch_{count}', k_ss)
    # convert images to float as attack requires to take gradients wrt to them

    if (FLAGS.adv_train):
        adversarial_images_train = pgd_attack(
            images, labels, params, epsilon=FLAGS.epsilon)
    else:
        adversarial_images_train = images
    # train on adversarial images
    params, state = jitted_update(
        params=params,
        state=state,
        l2reg=FLAGS.l2reg,
        data=(adversarial_images_train, labels))

    # Once per epoch evaluate the model on train/test sets and save kernel.
    if state.iter_num % iter_per_epoch_train == iter_per_epoch_train - 1:

      # compute train set accuracy, both on clean and adversarial images
      adversarial_accuracy_train_sample = 0.
      accuracy_train_sample = 0.
      for i in range(iter_per_epoch_train):
        images, labels = next(train_ds)
        images = images.astype(jnp.float32) / 255
        if (LINEARIZED):
            accuracy_train_sample += jnp.mean(lin_accuracy(params, (images, labels))) / iter_per_epoch_train
            adversarial_images_train = lin_pgd_attack(
              images, labels, params, epsilon=FLAGS.epsilon)
        else:
            accuracy_train_sample += jnp.mean(accuracy(params, (images, labels))) / iter_per_epoch_train
            adversarial_images_train = pgd_attack(
              images, labels, params, epsilon=FLAGS.epsilon)
        if (i == 0):
            # if (it > 2*iter_per_epoch_train):
            #     assert all(prev == labels[:5])
            if (it > 2*iter_per_epoch_train):
                print(prev, labels[:5])
            count += 1
            jnp.save(path+f'/samples/images_epoch_{count}', adversarial_images_train)
            jnp.save(path+f'/samples/labels_epoch_{count}', labels)
            with open(path+f'/samples/params_epoch_{count}.pickle', 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # jnp.save(path+f'/samples/params_epoch_{count}', params)
            if (count < 50):
                k_ss = kernel_fn(adversarial_images_train, None, params)
                jnp.save(path+f'/kernels/kernel_epoch_{count}', k_ss)
            prev = labels[:5]
        if (LINEARIZED):
            adversarial_accuracy_train_sample += jnp.mean(lin_accuracy(params, (adversarial_images_train, labels))) / iter_per_epoch_train
        else:
            adversarial_accuracy_train_sample += jnp.mean(accuracy(params, (adversarial_images_train, labels))) / iter_per_epoch_train
      accuracy_train.append(accuracy_train_sample)
      adversarial_accuracy_train.append(adversarial_accuracy_train_sample)

      # compute train set accuracy, both on clean and adversarial images
      adversarial_accuracy_test_sample = 0.
      accuracy_test_sample = 0.
      for _ in range(iter_per_epoch_test):
        images, labels = next(test_ds)
        images = images.astype(jnp.float32) / 255
        if (LINEARIZED):
            accuracy_test_sample += jnp.mean(lin_accuracy(params, (images, labels))) / iter_per_epoch_test
            adversarial_images_test = lin_pgd_attack(
              images, labels, params, epsilon=FLAGS.epsilon)
            adversarial_accuracy_test_sample += jnp.mean(lin_accuracy(params, (adversarial_images_test, labels))) / iter_per_epoch_test
        else:
            accuracy_test_sample += jnp.mean(accuracy(params, (images, labels))) / iter_per_epoch_test
            adversarial_images_test = pgd_attack(
              images, labels, params, epsilon=FLAGS.epsilon)
            adversarial_accuracy_test_sample += jnp.mean(accuracy(params, (adversarial_images_test, labels))) / iter_per_epoch_test
      accuracy_test.append(accuracy_test_sample)
      adversarial_accuracy_test.append(adversarial_accuracy_test_sample)


      time_elapsed = (datetime.datetime.now().replace(microsecond=0) - start)
      print(f"Epoch {it // iter_per_epoch_train} out of {FLAGS.epochs}")
      print(f"Accuracy on train set: {accuracy_train[-1]:.3f}")
      print(f"Accuracy on test set: {accuracy_test[-1]:.3f}")
      print(
          f"Adversarial accuracy on train set: {adversarial_accuracy_train[-1]:.3f}"
      )
      print(
          f"Adversarial accuracy on test set: {adversarial_accuracy_test[-1]:.3f}"
      )
      print(f"Time elapsed: {time_elapsed}")
      print()

  jnp.save(path+'/results/train_acc', accuracy_train)
  jnp.save(path+'/results/test_acc', accuracy_test)
  jnp.save(path+'/results/train_adv_acc', adversarial_accuracy_train)
  jnp.save(path+'/results/test_adv_acc', adversarial_accuracy_test)
  if (FLAGS.adv_train):
      plt.title(f"Adversarial training on {FLAGS.dataset}")
  else:
      plt.title(f"Natural training on {FLAGS.dataset}")
  plt.plot(accuracy_train, lw=3, label="clean accuracy on train set." , marker='<')
  plt.plot(accuracy_test, lw=3, label="clean accuracy on test set.", marker='d')
  plt.plot(
      adversarial_accuracy_train,
      lw=3,
      label="adversarial accuracy on train set.", marker='^')
  plt.plot(
      adversarial_accuracy_test,
      lw=3,
      label="adversarial accuracy on test set.", marker='>')
  plt.grid()
  plt.legend(frameon=False, ncol=2)
  plt.savefig(path+'/results/train_curves.pdf')
  plt.show()



if __name__ == "__main__":
  app.run(main)
