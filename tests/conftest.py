"""General config for tests"""

__author__ = "Guillaume Genthial"

import pytest


@pytest.fixture
def generator_fn():

    def gen():
        yield ([0, 1, 0, 0, 0, 2, 3, 0, 0, 1],
               [0, 1, 0, 0, 1, 2, 0, 3, 3, 1])
        yield ([0, 1, 0, 1, 0, 2, 3, 0, 0, 0],
               [0, 1, 0, 2, 1, 2, 0, 3, 1, 0])

    return gen


@pytest.fixture
def y_true_all(generator_fn):
    y_true_all = []
    for y_true, _ in generator_fn():
        y_true_all.extend(y_true)
    return y_true_all


@pytest.fixture
def y_pred_all(generator_fn):
    y_pred_all = []
    for _, y_pred in generator_fn():
        y_pred_all.extend(y_pred)
    return y_pred_all
