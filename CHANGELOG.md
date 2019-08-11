# Changelog

## v0.4.0 (Aug 10, 2019)

### Features
- Add `io.round_steps` and `io.round_step` functions
    ([312aef7](https://github.com/manifest/machine-learning-toolset/commit/312aef7a6df12d0dc21f40785f75c022e49f6800))
- Add `io.to_tfdata` function
    ([668041b](https://github.com/manifest/machine-learning-toolset/commit/668041b9ce00ec7c49d2f88d381960ff4e452695))

### Changes
- Introduce `tf.adapter` and `tf.generator` modules. Move tensorflow specific code under the `tf` module.
    ([6134e22](https://github.com/manifest/machine-learning-toolset/commit/6134e2237ca0a0f97fa5f4a956fda2a7b56e38ab))
- Replace `ModelAdapter` class with `Adaper`. Add `AdapterOptions` class.
    ([a7cc859](https://github.com/manifest/machine-learning-toolset/commit/a7cc85934b3df2f52ef36455018365d5a62aec05))
- Use `tf.data` internally in the `ModelAdapter`
    ([4680048](https://github.com/manifest/machine-learning-toolset/commit/468004817dcb38e32e1fbbb7ef540a10ed0f8ee8))
- Wrap datasets in the `io.normilize` output into a tuple. Allow to reshape all datasets at once with `io.reshape`.
    ([21516fd](https://github.com/manifest/machine-learning-toolset/commit/21516fd7734331557122560510518b6b19e90fa6))
- Replace `estimate_parameters` function of the `ModelAdapter` with `optimization` and rearrange arguments of `build_model`, `loss`, and analyze functions
    ([0ebac6c](https://github.com/manifest/machine-learning-toolset/commit/0ebac6c96418a929181c13a734c23a2cde983864))



## v0.3.0 (Aug 6, 2019)

### Features
- Add `generalized_metrics` funnction to the `ModelAdapter`
    ([f54070](https://github.com/manifest/machine-learning-toolset/commit/f5407072da5b34473770f38a5f0c1246d5d96c26))
- Add `io.reshape` function and support of target shape in `io.normalize`
    ([b705127](https://github.com/manifest/machine-learning-toolset/commit/b70512716bb970e2b89f81863fc3e610b07551b9)) 
- Add `save_hyperparameters` and `load_hyperparameters` functions of the `tf.ModelAdapter`
    ([4789c31](https://github.com/manifest/machine-learning-toolset/commit/4789c315cd091b084816d33acf8a91139e67b93f))

### Changes
- Pass options to the `build_model` function of the `tf.ModelAdapter`
    ([7c7d4fe](https://github.com/manifest/machine-learning-toolset/commit/7c7d4fea318fc8a1d245a2d515d8e2177f7f1f58))



## v0.2.0 (Aug 5, 2019)

### Features
- Add `io.slice` function
    ([ce6d221](https://github.com/manifest/machine-learning-toolset/commit/ce6d221c78c88669759b212bcb1ca9cccf1b95b5))
- Add `metric`, `metrics_history`, and `tune_hyperparameters` methods to the `tf.ModelAdapter`
    ([8787aa](https://github.com/manifest/machine-learning-toolset/commit/48787aaea3b1f7475b04386ee94d2b47b32a25e3), [808fc98](https://github.com/manifest/machine-learning-toolset/commit/808fc98b3b255ce762f5fbd1a4fd0f98841b6534))
- Add support of random seed for the `tf.ModelAdapter`
    ([9736382](https://github.com/manifest/machine-learning-toolset/commit/973638221e3c9da7d33ab3a2745e22667275dbfa))
- Add `tf.LambdaRangeGenerator` class
    ([808fc98](https://github.com/manifest/machine-learning-toolset/commit/808fc98b3b255ce762f5fbd1a4fd0f98841b6534))
- Add `tf.RangeGenerator` class
    ([808fc98](https://github.com/manifest/machine-learning-toolset/commit/808fc98b3b255ce762f5fbd1a4fd0f98841b6534))

### Changes
- Change arguments and names of the `tf.ModelAdapter` methods
    ([08d797a](https://github.com/manifest/machine-learning-toolset/commit/08d797a8b12c14a3ba25924e7b4b5fe71086a892), [7c017bd](https://github.com/manifest/machine-learning-toolset/commit/7c017bd0a7ae56a2434a65e6a11cff7e89faeb7b))



## v0.1.0 (Jul 21, 2019)

Initial release
