# TF Metrics

[![Build Status](https://travis-ci.org/guillaumegenthial/tf_metrics.svg?branch=master)](https://travis-ci.org/guillaumegenthial/tf_metrics)

Multi-class metrics for Tensorflow.

## Install

```
pip install -r requirements.txt
```


## Example

Pre-requisite: understand the general `tf.metrics` API. See for instance [the official guide on custom estimators](https://www.tensorflow.org/guide/custom_estimators#evaluate) or the [official documentation](https://www.tensorflow.org/api_docs/python/tf/metrics/accuracy).


Simple example

```python
import tensorflow as tf
import tf_metrics

y_true = [0, 1, 0, 0, 0, 2, 3, 0, 0, 1]
y_pred = [0, 1, 0, 0, 1, 2, 0, 3, 3, 1]
pos_indices = [1, 2, 3]  # Class 0 is the 'negative' class
num_classes = 4
average = 'micro'

# Tuple of (value, update_op)
precision = tf_metrics.precision(
    y_true, y_pred, num_classes, pos_indices, average=average)
recall = tf_metrics.recall(
    y_true, y_pred, num_classes, pos_indices, average=average)
f2 = tf_metrics.fbeta(
    y_true, y_pred, num_classes, pos_indices, average=average, beta=2)
f1 = tf_metrics.f1(
    y_true, y_pred, num_classes, pos_indices, average=average)

# Run the update op and get the updated value
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(precision[1])
```


If you want to use it with `tf.estimator.Estimator`, add to your `model_fn`


```python
metrics = {
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'f2': f2
    }
# For Tensorboard
for metric_name, metric in metrics.items():
    tf.summary.scalar(metric_name, metric[1])

if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=metrics)
```
