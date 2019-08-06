from . import normalization as _norm
import pandas as pd
import numpy as np
from functools import reduce
import math

def merge(data_acc):
    """Merge multiple dataframes into one."""
    if type(data_acc) != list: raise TypeError("'data_acc' isn't a list")
    if len(data_acc) < 1: raise ValueError("'data_acc' is an empty list")

    return reduce(lambda acc, data: acc.merge(data), data_acc[1:], data_acc[0])

def check(data):
    """Check the dataframe for undefined values."""
    if type(data) != pd.core.frame.DataFrame: raise TypeError("'data' isn't a DataFrame")
    if sum(data.isna().sum()) != 0: raise ValueError("Some elements of the dataset are undefined")

def split(data, ycol, shape, seed = None):
    """Split the specified dataset into training, development, and testing. Randomly shuffle training examples."""
    if type(data) != pd.core.frame.DataFrame: raise TypeError("'data' isn't a DataFrame")
    if type(ycol) != int: raise TypeError("'ycol' isn't an int")

    m = data.shape[0]
    if type(shape) == int:
        rest = (m - shape) / 2
        shape = (shape, math.ceil(rest), math.floor(rest))
    if len(shape) != 3: raise RuntimeError("'shape' isn't a 3-element list")

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
    """Retrieve a slice of the dataset."""
    if type(ds) != tuple: raise TypeError("'ds' isn't a tuple")
    if len(ds) != 2: raise ValueError("'ds' isn't a 2-element tuple")
    if idx_stop < idx_start: raise RuntimeError("'idx_start' value of the range has a greater value then 'idx_stop'")

    return (
        ds[0][idx_start:idx_stop, :],
        ds[1][idx_start:idx_stop, :],
    )

def reshape(ds, shape):
    """Reshape input in the dataset."""
    if type(shape) != tuple: raise TypeError("'shape' isn't a tuple")
    if type(ds) != tuple: raise TypeError("'ds' isn't a tuple")
    if len(ds) != 2: raise ValueError("'ds' isn't a 2-element tuple")

    return (
        ds[0].reshape((-1, *shape)),
        ds[1],
    )

def normalize(dss, target_shape=None):
    """Estimate normalization parametes on the training dataset and apply them to development and testing datasets."""
    if type(dss) != tuple: raise TypeError("'dss' isn't a tuple")
    if len(dss) != 3: raise ValueError("'dss' isn't a 3-element tuple")

    (X_train_in, y_train), (X_dev_in, y_dev), (X_test_in, y_test) = dss

    mu, sigma = _norm.estimate(X_train_in)
    X_train = _norm.apply(X_train_in, mu, sigma)
    X_dev = _norm.apply(X_dev_in, mu, sigma)
    X_test = _norm.apply(X_test_in, mu, sigma)

    dss_norm = (
        (X_train, y_train),
        (X_dev, y_dev),
        (X_test, y_test),
    )
    if (target_shape != None):
        dss_norm = (
            reshape(dss_norm[0], target_shape),
            reshape(dss_norm[1], target_shape),
            reshape(dss_norm[2], target_shape),
        )

    return (
        *dss_norm,
        mu,
        sigma
    )
