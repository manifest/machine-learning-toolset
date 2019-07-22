import pandas as pd
import numpy as np

class ModelAdapter():
    """Abstract base class for model adapters."""
    
    def __init__(self, hparams):
        model = self.build_model(hparams=hparams)
        
        self.model = model
        self.hparams = hparams
    
    def fit(self, ds, epochs):
        self.model = self.estimate_parameters(self.hparams, self.model, ds, epochs)
        
    def evaluate(self, ds):
        X, y = ds
        return self.model.evaluate(X, y)

    def analyze_training_examples(self, ds_train, ds_dev, epochs):
        return _analyze_training_examples(
            self.build_model,
            lambda hparams, ds: self.estimate_parameters(
                hparams,
                self.build_model(hparams, metrics=[]),
                ds,
                epochs=epochs,
            ).get_weights(),
            self.hparams,
            ds_train,
            ds_dev
        )
    
    @staticmethod
    def build_model(hparams, metrics = []):
        """Build the model."""

        raise NotImplementedError()
        
    @staticmethod
    def estimate_parameters(hparams, model, ds, epochs, callbacks=[]):
        """Estimate parameters of the model."""

        raise NotImplementedError()

def _evaluate_cost(build_model, hparams, Theta, ds):
    """Apply forward propagation with the specified parameters and estimate the cost of the learning algorithm."""

    model = build_model(hparams=hparams, metrics=[])
    X, y = ds

    model.call(X)
    model.set_weights(Theta)
    j = model.evaluate(X, y)

    return j

def _analyze_training_examples(build_model, optimize, hparams, ds_train, ds_dev, steps=10):
    """Use model selection algorithm to check for underfitting problem."""

    X_train, y_train = ds_train
    X_dev, y_dev = ds_dev
    m, n_x = X_train.shape

    hparams_without_regularization = hparams.copy()
    hparams_without_regularization["lambda"] = 0.
    
    m_acc = np.linspace(start=0, stop=m, num=steps, dtype=int)
    hist = pd.DataFrame(columns = ["m", "E_train", "E_dev"])

    for i in m_acc:
        m_train_slice = i + 1
        X_train_slice = X_train[0:m_train_slice, :]
        y_train_slice = y_train[0:m_train_slice]

        Theta = optimize(hparams, (X_train_slice, y_train_slice))
        E_train = _evaluate_cost(build_model, hparams_without_regularization, Theta, (X_train, y_train))
        E_dev = _evaluate_cost(build_model, hparams_without_regularization, Theta, (X_dev, y_dev))

        hist = hist.append(
            pd.DataFrame({"m": m_train_slice, "E_train": E_train, "E_dev": E_dev}, index = [0]),
            ignore_index=True
        )

    return hist
