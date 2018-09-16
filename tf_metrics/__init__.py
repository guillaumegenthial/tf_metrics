"""Multiclass"""

__author__ = "Guillaume Genthial"

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix


def precision(labels, predictions, num_classes, pos_indices, weights=None):
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    pr, _, _ = metrics_from_confusion_matrix(cm, num_classes, pos_indices)
    op, _, _ = metrics_from_confusion_matrix(op, num_classes, pos_indices)
    return (pr, op)


def recall(labels, predictions, num_classes, pos_indices, weights=None):
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    _, re, _ = metrics_from_confusion_matrix(cm, num_classes, pos_indices)
    _, op, _ = metrics_from_confusion_matrix(op, num_classes, pos_indices)
    return (re, op)


def f1(labels, predictions, num_classes, pos_indices, weights=None, beta=1):
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    _, _, f1 = metrics_from_confusion_matrix(cm, num_classes, pos_indices)
    _, _, op = metrics_from_confusion_matrix(op, num_classes, pos_indices)
    return (f1, op)


def safe_div(numerator, denominator):
    """Safe division, return 0 if denominator is 0"""
    numerator, denominator = tf.to_float(numerator), tf.to_float(denominator)
    zeros = tf.zeros_like(numerator, dtype=numerator.dtype)
    denominator_is_zero = tf.equal(denominator, zeros)
    return tf.where(denominator_is_zero, zeros, numerator / denominator)


def metrics_from_confusion_matrix(cm, num_classes, pos_indices, beta=1):
    neg_indices = [i for i in range(num_classes) if i not in pos_indices]
    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, neg_indices] = 0
    diag_sum = tf.reduce_sum(tf.diag_part(cm * cm_mask))

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[:, neg_indices] = 0
    tot_pred = tf.reduce_sum(cm * cm_mask)

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, :] = 0
    tot_gold = tf.reduce_sum(cm * cm_mask)

    pr = safe_div(diag_sum, tot_pred)
    re = safe_div(diag_sum, tot_gold)
    f1 = safe_div((1. + beta**2) * pr * re, beta**2 * pr + re)

    return pr, re, f1
