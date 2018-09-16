# pylint: disable=redefined-outer-name
"""Tests for """

__author__ = "Guillaume Genthial"

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import precision_score
from tensorflow.errors import OutOfRangeError

from tf_metrics import precision


@pytest.mark.parametrize("pos_indices", [
    [0, 1, 2],
    [0, 1, 3],
    [0]
])
@pytest.mark.usefixtures('generator_fn')
def test_precision(pos_indices, generator_fn):
    for y_true, y_pred in generator_fn():
        pr_tf = precision(y_true, y_pred, 4, pos_indices)
        pr_sk = precision_score(y_true, y_pred, pos_indices, average='micro')
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            assert np.allclose(sess.run(pr_tf[1]), pr_sk)


@pytest.mark.parametrize("pos_indices", [
    [0, 1, 2],
    [0, 1, 3],
    [0]
])
@pytest.mark.usefixtures('generator_fn', 'y_true_all', 'y_pred_all')
def test_precision_op(pos_indices, generator_fn, y_true_all, y_pred_all):
    # Precision on the whole dataset
    pr_sk = precision_score(
        y_true_all, y_pred_all, pos_indices, average='micro')

    # Create Tensorflow graph
    ds = tf.data.Dataset.from_generator(
        generator_fn, (tf.int32, tf.int32), ([None], [None]))
    y_true, y_pred = ds.make_one_shot_iterator().get_next()
    pr_tf = precision(y_true, y_pred, 4, pos_indices)

    with tf.Session() as sess:
        # Initialize and run the update op on each batch
        sess.run(tf.local_variables_initializer())
        while True:
            try:
                sess.run(pr_tf[1])
            except OutOfRangeError as e:
                break

        # Check final value
        assert np.allclose(sess.run(pr_tf[0]), pr_sk)
