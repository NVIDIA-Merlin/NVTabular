from packaging import version
import tensorflow as tf


def configure_tensorflow(memory_allocation=None, device=None):
    # TODO: we have to do total now right?
    free_gpu_mem_mb = device_mem_size(kind="free") / (1024 ** 2)
    if memory_allocation is None:
        memory_allocation = os.environ.get("TF_MEMORY_ALLOCATION", 0.5)

    if float(memory_allocation) < 1:
        memory_allocation = free_gpu_mem_mb * float(memory_allocation)
    memory_allocation = int(memory_allocation)
    assert memory_allocation < free_gpu_mem_mb

    # TODO: what will this look like in any sort
    # of distributed set up?
    if device is None:
        device = os.environ.get("TF_VISIBLE_DEVICE", 0)
    try:
        tf.config.set_logical_device_configuration(
            tf.config.list_physical_devices("GPU")[device],
            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_allocation)],
        )
    except RuntimeError:
        warnings.warn("TensorFlow runtime already initialized, may not be enough memory for cudf")

    # versions using TF earlier than 2.3.0 need to use extension
    # library for dlpack support to avoid memory leak issue
    __TF_DLPACK_STABLE_VERSION = "2.3.0"
    if version.parse(tf.__version__) < version.parse(__TF_DLPACK_STABLE_VERSION):
        try:
            from tfdlpack import from_dlpack
        except ModuleNotFoundError as e:
            message = "If using TensorFlow < 2.3.0, you must install tfdlpack-gpu extension library"
            raise ModuleNotFoundError(message) from e

    else:
        from tensorflow.experimental.dlpack import from_dlpack

    return from_dlpack
