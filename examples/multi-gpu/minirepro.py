import random

import cudf
import tensorflow as tf
from tensorflow.experimental.dlpack import from_dlpack

# gpu_to_use = 0      # Works
gpu_to_use = 1  # Errors

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_visible_devices(gpus[gpu_to_use], "GPU")

tf.random.uniform((1,))  # need TF context initialized

gdf = cudf.DataFrame()
gdf["scalar"] = [random.uniform(0.0, 1.0) for i in range(10)]

dlpack = gdf.to_dlpack()
x = from_dlpack(dlpack)
