from .. import io as _io
import pandas as pd
import numpy as np
import tensorflow as tf
import json

class Options():
    def __init__(self, epochs=None, verbose=None, seed=None):
        self.epochs = 1 if epochs == None else epochs
        self.verbose = 0 if verbose == None else verbose
        self.seed = seed

class Model():
    """A base class for a model adapter that provides methods for error analysis and hyperparameters tuning. Derived classes must implement methods creating the model and estimating parameters of the learning algorithm."""

    def __init__(self, options={}, hparams={}):
        self.options = options

        # There can't be any presets or default values for hyperparameters since we can't predict
        # all possible combinations of them or future algorithms they will be used in.
        self.hparams = hparams
        self.m = self.model(self.options, self.hparams)

    def fit(self, ds, metrics=[], callbacks=[]):
        """Estimate parameters of the model using a method provided by a derived class."""

        self.m = self.optimization(self.options, self.hparams, self.m, ds, metrics, callbacks)

    def evaluate(self, ds):
        """Estimate the loss and metrics values for the model."""

        return self.m.evaluate(ds, verbose=self.options.verbose)

    def tune_hyperparameters(self, gen, ds_train, ds_dev, deep=10):
        """Use model selection algorithm to find the best hyperparameters values. Update hyperparameters values of the adapter."""

        result = self.analyze_hyperparameters(self.options, self.hparams, gen, ds_train, ds_dev, deep)
        self.hparams = {**self.hparams, **_read_hyperparameters(*result)}
        return result

    @classmethod
    def analyze_hyperparameters(Self, options, hparams, gen, ds_train, ds_dev, deep=10):
        """Use model selection algorithm to find the best hyperparameters values."""

        return _analyze_hyperparameters_deep(
            Self.model,
            Self.loss(options),
            lambda _hparams, _ds: Self.optimization(
                options,
                _hparams,
                Self.model(options, _hparams),
                _ds,
                metrics=[],
                callbacks=[],
            ).get_weights(),
            options,
            hparams,
            gen,
            ds_train,
            ds_dev,
            deep,
        )

    @classmethod
    def analyze_dataset(Self, options, hparams, ds_train, ds_dev, step_size = None, n_steps = 10):
        """Use model selection algorithm to determine if the dataset has an underfitting problem."""
        if n_steps < 1: raise ValueError("'n_steps' is less then 1")

        ## Note that we use experimental functional here
        ## to estimate a 'step_size' that cover the entire dataset the best in 'n_steps'
        ## [the use is optional, but affects the function interface]
        if step_size == None:
            step_size = _io.round_step(tf.data.experimental.cardinality(ds_train).numpy(), n_steps)
        if step_size < 1: raise ValueError("'step_size' is less then 1")

        return _analyze_dataset(
            Self.model,
            Self.loss(options),
            lambda _hparams, _ds: Self.optimization(
                options,
                _hparams,
                Self.model(options, _hparams),
                _ds,
                metrics=[],
                callbacks=[],
            ).get_weights(),
            options,
            hparams,
            ds_train,
            ds_dev,
            step_size,
            n_steps,
        )

    def load_hyperparameters(self, path):
        """Load hyperparameters from the JSON file."""

        with open(path, 'r') as file:
            self.hparams = json.load(file)

    def save_hyperparameters(self, path):
        """Save hyperparameters to the JSON file."""

        with open(path, 'w') as file:
            json.dump(self.hparams, file)

    def metric(self, name):
        "Retrieve the last value of metric with specified name from the model history."

        return self.m.history.history[name][-1]

    def metrics_history(self):
        "Retrieve historical metric values."

        return pd.DataFrame(self.m.history.history).rename_axis("epoch").reset_index()

    def generalized_metrics(self, ds_dev, ds_test):
        "Evaluate the model on development and testing sets and return the metrics along with metrics of the training set."

        metrics_dev = self.evaluate(ds_dev)
        metrics_test = self.evaluate(ds_test)
        keys = list(self.m.history.history.keys())
        return pd.DataFrame(
            list(map(
                lambda i: dict(train=self.metric(keys[i]), dev=metrics_dev[i], test=metrics_test[i]),
                range(len(keys))
            )),
            index=keys,
        )

    @staticmethod
    def model(options, hparams):
        """Build and return the model."""

        raise NotImplementedError()

    @staticmethod
    def loss(options):
        """Return the loss function."""

        raise NotImplementedError()

    @staticmethod
    def optimization(options, hparams, model, ds, metrics=[], callbacks=[]):
        """Estimate parameters of the learning algorithm and return the updated model."""

        raise NotImplementedError()

def _evaluate_loss(model, loss, options, hparams, Theta, ds):
    """Apply forward propagation with the specified parameters to a newly created model and estimate the cost of the learning algorithm."""

    m = model(options, hparams)
    ## We don't need any optimizer to just evaluate the loss.
    ## We specify the stochastic gradient descent just because the optimizer argument is required by the compile function.
    m.compile(optimizer='sgd', loss=loss)
    m.set_weights(Theta)

    j = m.evaluate(ds, verbose=options.verbose)

    return j

def _analyze_dataset(model, loss, optimization, options, hparams, ds_train, ds_dev, step_size, n_steps, hist=pd.DataFrame()):
    """Use model selection algorithm to determine if the dataset has an underfitting problem."""

    hparams_without_regularization = _reset_regularization_hyperparameter(hparams)

    for step in range(n_steps):
        m_train_slice = step_size * step + 1
        ds_train_slice = ds_train.take(m_train_slice)
        ## Note that we use experimental functional here
        ## to break from the loop if 'ds_train' has been exhausted
        ## [the use is optional]
        if tf.data.experimental.cardinality(ds_train_slice).numpy() < m_train_slice:
            break

        ## We use regularization when estimating parameters
        Theta = optimization(hparams, ds_train_slice)
        ## We don't use regularization when computing the training and development error
        ## The training set error is computed on the training subset
        E_train = _evaluate_loss(model, loss, options, hparams_without_regularization, Theta, ds_train_slice)
        ## However, the cross validation error is computed over the entire development set
        E_dev = _evaluate_loss(model, loss, options, hparams_without_regularization, Theta, ds_dev)

        hist = hist.append(pd.DataFrame({"E_train": E_train, "E_dev": E_dev, "m_batch": m_train_slice}, index = [step]))

    return hist

def _analyze_hyperparameters_deep(model, loss, optimization, options, hparams, gen, ds_train, ds_dev, deep, hist=pd.DataFrame()):
    """Use model selection algorithm to find the best hyperparameters values utilizing a range generator."""

    modifiers = gen.generate()
    result = _analyze_hyperparameters(model, loss, optimization, options, hparams, modifiers, ds_train, ds_dev, hist)

    deep -= 1
    if deep < 1:
        return result

    idx, hist = result
    gen.adjast(_read_hyperparameters(idx, hist))
    return _analyze_hyperparameters_deep(model, loss, optimization, options, hparams, gen, ds_train, ds_dev, deep, hist)

def _analyze_hyperparameters(model, loss, optimization, options, hparams, modifiers, ds_train, ds_dev, hist=pd.DataFrame()):
    """Use model selection algorithm to find the best hyperparameters values on the specified range of values."""

    for modifier in modifiers:
        modified_hparams = {**hparams, **modifier}
        modified_hparams_without_regularization = _reset_regularization_hyperparameter(modified_hparams)

        ## We use regularization when estimating parameters
        Theta = optimization(modified_hparams, ds_train)
        ## We don't use regularization when computing the training and development error
        ## The training set error is computed on the training subset
        E_train = _evaluate_loss(model, loss, options, modified_hparams_without_regularization, Theta, ds_train)
        ## However, the cross validation error is computed over the entire development set
        E_dev = _evaluate_loss(model, loss, options, modified_hparams_without_regularization, Theta, ds_dev)

        hist = hist.append(
            pd.DataFrame({"E_train": E_train, "E_dev": E_dev, **modifier}, index = [0]),
            ignore_index=True
        )

    idx = hist["E_dev"].idxmin()
    return (idx, hist)

def _read_hyperparameters(idx, hist):
    """Read hyperparameters as a dictionary from the specified history dataset."""

    return hist.iloc[idx, 2:].to_dict()

def _reset_regularization_hyperparameter(hparams):
    """Reset regularization hyperparameter lambda."""

    modified_hparams = hparams.copy()
    if "lambda" in modified_hparams:
        del modified_hparams["lambda"]
    return modified_hparams
