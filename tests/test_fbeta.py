"""Tests for f-beta score"""

__author__ = "Guillaume Genthial"

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import fbeta_score
from tensorflow.errors import OutOfRangeError

import tf_metrics


@pytest.mark.usefixtures('generator_fn')
@pytest.mark.parametrize("pos_indices", [
    [0, 1, 2],
    [0, 1, 3],
    [0],
    [0, 1, 2, 3],
    None
])
@pytest.mark.parametrize("average", [
    'macro', 'micro', 'weighted'
])
@pytest.mark.parametrize("beta", [
    1, 2
])
def test_fbeta(generator_fn, pos_indices, average, beta):
    for y_true, y_pred in generator_fn():
        pr_tf = tf_metrics.fbeta(
            y_true, y_pred, 4, pos_indices, average=average, beta=beta)
        pr_sk = fbeta_score(
            y_true, y_pred, beta, pos_indices, average=average)
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            assert np.allclose(sess.run(pr_tf[1]), pr_sk)


@pytest.mark.usefixtures('generator_fn', 'y_true_all', 'y_pred_all')
@pytest.mark.parametrize("pos_indices", [
    [0, 1, 2],
    [0, 1, 3],
    [0],
    [0, 1, 2, 3],
    None
])
@pytest.mark.parametrize("average", [
    'macro', 'micro', 'weighted'
])
@pytest.mark.parametrize("beta", [
    1, 2
])
def test_fbeta_op(generator_fn, y_true_all, y_pred_all, pos_indices,
                  average, beta):
    # Precision on the whole dataset
    pr_sk = fbeta_score(
        y_true_all, y_pred_all, beta, pos_indices, average=average)

    # Create Tensorflow graph
    ds = tf.data.Dataset.from_generator(
        generator_fn, (tf.int32, tf.int32), ([None], [None]))
    y_true, y_pred = ds.make_one_shot_iterator().get_next()
    pr_tf = tf_metrics.fbeta(y_true, y_pred, 4, pos_indices,
                             average=average, beta=beta)

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
