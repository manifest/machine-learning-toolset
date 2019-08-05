from . import normalization as _norm
import pandas as pd
import numpy as np
from functools import reduce
import math

def merge(data_acc):
    """Merge multiple dataframes into one."""

    assert len(data_acc) > 0, "at least one dataframe is provided"
    return reduce(lambda acc, data: acc.merge(data), data_acc[1:], data_acc[0])

def check(data):
    """Check the dataframe for undefined values."""

    assert type(data) == pd.core.frame.DataFrame
    assert sum(data.isna().sum()) == 0, "all elements of the dataset are defined"

def split(data, ycol, shape, seed = 1):
    """Split the specified dataset into training, development, and testing. Randomly shuffle training examples."""

    assert type(data) == pd.core.frame.DataFrame
    assert type(ycol) == int
    m = data.shape[0]
    if type(shape) == int:
        rest = (m - shape) / 2
        shape = (shape, math.ceil(rest), math.floor(rest))
    assert len(shape) == 3

    # Randomly shuffle training examples in the dataset
    data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Separate labels from features
    n_x = len(data.columns) - 1
    ds = (
        data.drop(data.columns[ycol], axis = 1).to_numpy(dtype=float).reshape((-1, n_x)),
        data.iloc[:, ycol].to_numpy(dtype=float).reshape((-1, 1)),
    )

    # Split the dataset into training, development, and testing
    m_train, m_dev, m_test = shape
    idx_train = m_train
    idx_dev = idx_train + m_dev
    idx_test = idx_dev + m_test

    return (
        slice(ds, 0, idx_train),
        slice(ds, idx_train, idx_dev),
        slice(ds, idx_dev, idx_test),
    )

def slice(ds, idx_start, idx_stop):
    """Retrieve a slice of the dataset"""

    return (ds[0][idx_start:idx_stop, :], ds[1][idx_start:idx_stop, :])

def normalize(ds):
    """Estimate normalization parametes on the training dataset and apply them to development and testing datasets."""

    assert len(ds) == 3
    (X_train_in, y_train), (X_dev_in, y_dev), (X_test_in, y_test) = ds

    mu, sigma = _norm.estimate(X_train_in)
    X_train = _norm.apply(X_train_in, mu, sigma)
    X_dev = _norm.apply(X_dev_in, mu, sigma)
    X_test = _norm.apply(X_test_in, mu, sigma)

    return (
        (X_train, y_train),
        (X_dev, y_dev),
        (X_test, y_test),
        mu,
        sigma
    )
