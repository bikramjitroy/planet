# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from planet import tools

slim = tf.contrib.slim


def input_network(weights_initializer, state):
  all_net = tf.cast(state, tf.float32)
  batch_size = all_net.get_shape().as_list()[0]
  stack_size = all_net.get_shape().as_list()[1]

  velocity_net = tf.slice(all_net, [0,0,0], [batch_size,stack_size, 242])
  velocity_net = tf.reshape(velocity_net, [-1, 2, 11, 11])
  velocity_net = slim.conv2d(velocity_net, 32, [4,4], 3, weights_initializer=weights_initializer)
  velocity_net = slim.conv2d(velocity_net, 64, [4,4], 2, weights_initializer=weights_initializer)
  velocity_net = slim.conv2d(velocity_net, 64, [4,4], 2, weights_initializer=weights_initializer)

  velocity_net = slim.flatten(velocity_net)
  velocity_net = slim.fully_connected(velocity_net, 16)

  body_net = tf.slice(all_net, [0,0,242], [batch_size,stack_size, 97])
  body_net = tf.reshape(body_net, [-1, 97])
  body_net = slim.fully_connected(body_net, 64)

  net = tf.concat([velocity_net, body_net], 1)
  net = slim.fully_connected(net, 128 , activation_fn=tf.nn.relu6)
  net = slim.fully_connected(net, 64 , activation_fn=tf.nn.relu6)
  return net


def encoder(obs):
  """Extract deterministic features from an observation."""

  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  hidden = input_network(weights_initializer, obs['state'])

  hidden = tf.reshape(hidden, tools.shape(obs['state'])[:2] + [
       np.prod(hidden.shape[1:].as_list())])

  return hidden

  # kwargs = dict(strides=2, activation=tf.nn.relu)
  # hidden = tf.reshape(obs['image'], [-1] + obs['image'].shape[2:].as_list())
  # hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs)
  # hidden = tf.layers.conv2d(hidden, 64, 4, **kwargs)
  # hidden = tf.layers.conv2d(hidden, 128, 4, **kwargs)
  # hidden = tf.layers.conv2d(hidden, 256, 4, **kwargs)
  # hidden = tf.layers.flatten(hidden)
  # assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()
  # hidden = tf.reshape(hidden, tools.shape(obs['image'])[:2] + [
  #     np.prod(hidden.shape[1:].as_list())])
  # return hidden


def decoder(state, data_shape):
  observation_size = 339
  """Compute the data distribution of an observation from its state."""
  hidden = slim.fully_connected(state, 1024 , activation_fn=tf.nn.relu6)
  hidden = slim.fully_connected(hidden, 512 , activation_fn=tf.nn.relu6)
  hidden = slim.fully_connected(hidden, 512 , activation_fn=tf.nn.relu6)
  hidden = slim.fully_connected(hidden, observation_size , activation_fn=tf.nn.relu6)
  hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1].value])

  mean = hidden
  assert mean.shape[1:].as_list() == [1, 1, observation_size], mean.shape
  mean = tf.reshape(mean, tools.shape(state)[:-1] + data_shape)
  dist = tfd.Normal(mean, 1.0)
  dist = tfd.Independent(dist, len(data_shape))
  return dist
