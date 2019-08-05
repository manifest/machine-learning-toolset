import pandas as pd
import numpy as np

class RangeGenerator():
    """Generate and narrow a range within specified boundaries."""

    def __init__(self, start, stop, steps, dtype=float):
        self.start = start
        self.stop = stop
        self.n_steps = steps
        self.dtype = dtype

        self._update_step()

    def generate(self):
        """Generate the range values."""

        return np.linspace(start=self.start, stop=self.stop, num=self.n_steps, dtype=self.dtype)
    
    def adjast(self, value):
        """Adjust the range to new narrow boundaries around the value."""

        self.start = max(value - self.step, self.start)
        self.stop = min(value + self.step, self.stop)

        self._update_step()
        
    def _update_step(self):
        if self.stop < self.start: raise RuntimeError("'start' value of the range has a greater value then 'stop'")

        self.step = (self.stop - self.start) / self.n_steps

class LambdaRangeGenerator(RangeGenerator):
    """Generate and narrow a range of lambda regularization hyperparameter values."""

    def __init__(self, start=0.0001, stop=10, steps=3, *args, **kwargs):
        super(LambdaRangeGenerator, self).__init__(start=start, stop=stop, steps=steps, *args, **kwargs)

    def adjast(self, hparam):
        if "lambda" not in hparam: raise ValueError("missing 'lambda' property of the 'hparam' dictionary")

        super().adjast(hparam["lambda"])

    def generate(self):
        return list(map(lambda v: {"lambda": v}, super().generate()))

class ModelAdapter():
    """A base class for a model adapter that provides methods for error analysis and hyperparameters tuning. Derived classes must implement methods creating the model and estimating parameters of the learning algorithm."""

    @staticmethod
    def verify_options(options):
        for value in ["epochs", "batch_size", "verbose", "seed"]:
            if value not in options: raise ValueError("missing '{}' property of the 'options' dictionary".format(value))

    def __init__(self, options={}, hparams={}, metrics=[]):
        self.options = {
            "epochs": 1,
            "batch_size": 32,
            "verbose": 0,
            "seed": None,
            **options,
        }

        # There can't be any presets or default values for hyperparameters since we can't predict
        # all possible combinations of them or future algorithms they will be used in.
        self.hparams = hparams
        self.model = self.build_model(self.hparams, metrics=metrics)

    def fit(self, ds):
        """Estimate parameters of the model using a method provided by a derived class."""

        self.model = self.estimate_parameters(self.options, self.model, ds)

    def evaluate(self, ds):
        """Estimate the loss and metrics values for the model."""

        X, y = ds
        return self.model.evaluate(X, y, verbose=self.options["verbose"])

    def analyze_dataset(self, ds_train, ds_dev, steps=10):
        """Use model selection algorithm to determine if the dataset has an underfitting problem."""

        return _analyze_dataset(
            self.build_model,
            lambda hparams, ds: self.estimate_parameters(
                self.options,
                self.build_model(hparams, metrics=[]),
                ds,
            ).get_weights(),
            self.hparams,
            ds_train,
            ds_dev,
            steps,
        )

    def tune_hyperparameters(self, gen, ds_train, ds_dev, deep=10):
        """Use model selection algorithm to find the best hyperparameters values. Update hyperparameters values of the adapter."""

        result = self.analyze_hyperparameters(gen, ds_train, ds_dev, deep)
        self.hparams = {**self.hparams, **_read_hyperparameters(*result)}
        return result

    def analyze_hyperparameters(self, gen, ds_train, ds_dev, deep=10):
        """Use model selection algorithm to find the best hyperparameters values."""

        return _analyze_hyperparameters_deep(
            self.build_model,
            lambda hparams, ds: self.estimate_parameters(
                self.options,
                self.build_model(hparams, metrics=[]),
                ds,
            ).get_weights(),
            self.hparams,
            gen,
            ds_train,
            ds_dev,
            deep,
        )

    def metric(self, name):
        "Retrieve the last value of metric with specified name from the model history."

        return self.model.history.history[name][-1]

    def metrics_history(self):
        "Retrieve historical metric values."

        return pd.DataFrame(self.model.history.history).rename_axis("epoch").reset_index()

    @staticmethod
    def build_model(hparams, metrics = []):
        """Implement the model."""

        raise NotImplementedError()

    @staticmethod
    def estimate_parameters(options, model, ds, callbacks=[]):
        """Implement the learning algorithm parameters estimation."""

        raise NotImplementedError()

def _evaluate_cost(build_model, hparams, Theta, ds):
    """Apply forward propagation with the specified parameters to a newly created model and estimate the cost of the learning algorithm."""

    model = build_model(hparams, metrics=[])
    X, y = ds

    model.call(X)
    model.set_weights(Theta)
    j = model.evaluate(X, y, verbose=0)

    return j

def _analyze_dataset(build_model, optimize, hparams, ds_train, ds_dev, steps, hist=pd.DataFrame()):
    """Use model selection algorithm to determine if the dataset has an underfitting problem."""

    X_train, y_train = ds_train
    m, n_x = X_train.shape

    m_acc = np.linspace(start=1, stop=m, num=steps, dtype=int)
    hparams_without_regularization = _reset_regularization_hyperparameter(hparams)

    for m_train_slice in m_acc:
        ds_train_slice = (X_train[0:m_train_slice, :], y_train[0:m_train_slice])

        ## We use regularization when estimating parameters
        Theta = optimize(hparams, ds_train_slice)
        ## We don't use regularization when computing the training and development error
        ## The training set error is computed on the training subset
        E_train = _evaluate_cost(build_model, hparams_without_regularization, Theta, ds_train_slice)
        ## However, the cross validation error is computed over the entire development set
        E_dev = _evaluate_cost(build_model, hparams_without_regularization, Theta, ds_dev)

        hist = hist.append(
            pd.DataFrame({"E_train": E_train, "E_dev": E_dev, "m": m_train_slice}, index = [0]),
            ignore_index=True
        )

    return hist

def _analyze_hyperparameters_deep(build_model, optimize, hparams, gen, ds_train, ds_dev, deep, hist=pd.DataFrame()):
    """Use model selection algorithm to find the best hyperparameters values utilizing a range generator."""

    modifiers = gen.generate()
    result = _analyze_hyperparameters(build_model, optimize, hparams, modifiers, ds_train, ds_dev, hist)

    deep -= 1
    if deep < 1:
        return result

    idx, hist = result
    gen.adjast(_read_hyperparameters(idx, hist))
    return _analyze_hyperparameters_deep(build_model, optimize, hparams, gen, ds_train, ds_dev, deep, hist)

def _analyze_hyperparameters(build_model, optimize, hparams, modifiers, ds_train, ds_dev, hist=pd.DataFrame()):
    """Use model selection algorithm to find the best hyperparameters values on the specified range of values."""

    X_train, y_train = ds_train
    m, n_x = X_train.shape

    for modifier in modifiers:
        modified_hparams = {**hparams, **modifier}
        modified_hparams_without_regularization = _reset_regularization_hyperparameter(modified_hparams)

        ## We use regularization when estimating parameters
        Theta = optimize(modified_hparams, ds_train)
        ## We don't use regularization when computing the training and development error
        ## The training set error is computed on the training subset
        E_train = _evaluate_cost(build_model, modified_hparams_without_regularization, Theta, ds_train)
        ## However, the cross validation error is computed over the entire development set
        E_dev = _evaluate_cost(build_model, modified_hparams_without_regularization, Theta, ds_dev)

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
