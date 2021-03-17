import random
import os
import tensorflow as tf
os.environ['TF_MEMORY_ALLOCATION'] = "0.4"

# gpu_to_use = 0      # Works
gpu_to_use = 1        # Errors

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.debugging.set_log_device_placement(True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus, "GPU")

print(gpus)
print(tf.config.list_physical_devices('GPU'))


import cupy
with tf.device("/GPU:" + str(gpu_to_use)):
    cupy.cuda.Device(gpu_to_use).use()
    # Converting from TF to CuPy with dlpack works for both devices
    tensor = tf.random.uniform((10,))

    dltensor = tf.experimental.dlpack.to_dlpack(tensor)
    array1 = cupy.fromDlpack(dltensor)

    # Converting from CuPy to TF with dlpack only works for device 0
    array1 = cupy.array([random.uniform(0.0, 1.0) for i in range(10)], dtype=cupy.float32)
    dltensor = array1.toDlpack()

    x = tf.experimental.dlpack.from_dlpack(dltensor)
