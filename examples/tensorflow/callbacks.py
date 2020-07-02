import tensorflow as tf
import time


class ThroughputLogger(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, window_size=10, **kwargs):
        self.batch_size = batch_size
        self.window_size = window_size
        self.times = []
        super(ThroughputLogger, self).__init__(**kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        self.times = [time.time()]

    def on_batch_end(self, batch, logs=None):
        self.times.append(time.time())
        if len(self.times) > self.window_size:
            del self.times[0]

        time_delta = self.times[-1] - self.times[0]
        tf.summary.scalar(
            'throughput', self.batch_size*len(self.times) / time_delta, step=batch)