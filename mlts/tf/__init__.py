from mlts.tf import adapter
from mlts.tf import generator

import tensorflow as tf

def to_dataset(dss, batch_size):
    """Convert to tf.data.Dataset"""

    return tuple(map(lambda ds: tf.data.Dataset.from_tensor_slices(ds).batch(batch_size), dss))
