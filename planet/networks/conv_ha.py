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
import keras
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from planet import tools


def encoder(obs):
  """Extract deterministic features from an observation."""
  obs = obs['image']
  hidden = tf.layers.dense(obs, 1024, tf.nn.relu)#keras.layers.Dense(500, activation='relu')(obs)
  hidden = tf.layers.dense(hidden, 1024, tf.nn.relu)#keras.layers.Dense(500, activation='relu')(hidden)
  hidden = tf.layers.dense(hidden, 1024, None)#keras.layers.Dense(1024)(hidden)
  return hidden


def decoder(state, data_shape):
  """Compute the data distribution of an observation from its state."""
  #hidden = keras.layers.Dense(500, activation='relu')(state)
  #hidden = keras.layers.Dense(500, activation='relu')(hidden)
  #hidden = keras.layers.Dense(26)(hidden)
  hidden = tf.layers.dense(state, 500, tf.nn.relu)
  hidden = tf.layers.dense(hidden, 500, tf.nn.relu)
  hidden = tf.layers.dense(hidden, 26, None)
  mean = hidden
  mean = tf.reshape(mean, tools.shape(state)[:-1] + data_shape)
  dist = tools.MSEDistribution(mean)
  dist = tfd.Independent(dist, len(data_shape))
  return dist
