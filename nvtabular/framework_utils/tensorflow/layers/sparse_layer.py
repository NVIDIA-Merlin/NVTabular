#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tensorflow as tf


class SparseTensor(tf.keras.layers.Layer):
    """
    Layer converts a a triplet of Tensors, index, row_lengths, values, into
    a Sparse Tensor (tf.sparse.SparseTensor).

    SparseTensor.call returns a tf.sparse.SparseTensor.

    Example usage::

        index = [[2],[5], [9], [7], [4], [3], [4],
                 [9], [3], [8], [1], [0], [3], [5]]
        row_lengths = [[2], [4], [4], [1], [3]]
        values = [[0.1], [3.4], [3.5], [5.3], [3. ], [1. ], [0.1],
                  [2. ], [0.4], [0.8], [0.5], [1. ], [2. ], [5. ]]

        tf_index = tf.convert_to_tensor(index)
        tf_row_lengths = tf.convert_to_tensor(row_lengths)
        tf_values = tf.convert_to_tensor(values)

        x = layers.SparseTensor(dense_dim=10)(tf_index, tf_row_lengths, tf_values)

    Parameters
    ----------
    dense_dim : int
        Second Dimension of the SparseTensor
    """

    def __init__(self, dense_dim, name=None, **kwargs):
        self.dense_dim = dense_dim
        super(SparseTensor, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, index, row_lengths, values):
        batch_size = tf.shape(row_lengths)[0]
        index = index[:, 0]
        row_lengths = row_lengths[:, 0]
        x = tf.RaggedTensor.from_row_lengths(index, row_lengths)
        dense_indices = x.to_tensor(default_value=-1)
        trange = tf.expand_dims(tf.range(tf.shape(dense_indices)[0]), 1)
        ind = tf.transpose(
            [
                tf.squeeze(
                    tf.expand_dims(
                        tf.repeat(trange, repeats=tf.shape(dense_indices)[1], axis=1), 1
                    ),
                    1,
                ),
                tf.cast(dense_indices, tf.int32),
            ]
        )
        ind = tf.reshape(tf.transpose(ind, [1, 0, 2]), shape=(-1, 2))
        mask = tf.where(ind[:, 1] != -1, 1, 0)
        x = tf.sparse.SparseTensor(
            indices=tf.cast(tf.boolean_mask(ind, mask), tf.int64),
            values=tf.squeeze(values[:, 0]),
            dense_shape=[batch_size, self.dense_dim],
        )
        return x

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], self.dense_dim)
