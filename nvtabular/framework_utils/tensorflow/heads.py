from collections import defaultdict
from typing import Optional, Dict, Text

import tensorflow as tf
from . import tfrs

from nvtabular.column_group import ColumnGroup, Tag
from .tfrs import Model


class Task(tfrs.Task):
    @classmethod
    def binary_classification(cls, metrics=None):
        metrics = metrics or [
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC()
        ]

        return tfrs.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )

    @classmethod
    def regression(cls, metrics=None):
        metrics = metrics or [
            tf.keras.metrics.RootMeanSquaredError()
        ]

        return tfrs.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=metrics
        )


class Head(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tasks = {}
        self._tasks_prepares = {}
        self._task_weights = defaultdict(lambda: 1)

    @classmethod
    def from_column_group(cls, column_group: ColumnGroup, add_logits=True, task_weights=None):
        if task_weights is None:
            task_weights = {}
        to_return = cls()

        for binary_target in column_group.get_tagged(Tag.TARGETS_BINARY).columns:
            to_return = to_return.add_binary_classification_task(binary_target, add_logit_layer=add_logits,
                                                                 task_weight=task_weights.get(binary_target, 1))

        for regression_target in column_group.get_tagged(Tag.TARGETS_REGRESSION).columns:
            to_return = to_return.add_regression_task(regression_target, add_logit_layer=add_logits,
                                                      task_weight=task_weights.get(regression_target, 1))

        # TODO: Add multi-class classification here. Figure out how to get number of classes

        return to_return

    def add_task(self, target_name, task: Task, pre: Optional[tf.keras.layers.Layer] = None, task_weight=1):
        self._tasks[target_name] = task
        if pre:
            self._tasks_prepares[target_name] = pre
        if task_weight:
            self._task_weights[target_name] = task_weight

        return self

    def add_binary_classification_task(self, target_name, add_logit_layer=True, task_weight=1):
        self._tasks[target_name] = Task.binary_classification()
        if add_logit_layer:
            self._tasks_prepares[target_name] = tf.keras.layers.Dense(1, activation="sigmoid",
                                                                      name=f"binary/{target_name}")
        if task_weight:
            self._task_weights[target_name] = task_weight

        return self

    def add_regression_task(self, target_name, add_logit_layer=True, task_weight=1):
        self._tasks[target_name] = Task.regression()
        if add_logit_layer:
            self._tasks_prepares[target_name] = tf.keras.layers.Dense(1, name=f"regression/{target_name}")
        if task_weight:
            self._task_weights[target_name] = task_weight

        return self

    def pop_labels(self, inputs: Dict[Text, tf.Tensor]):
        outputs = {}
        for name in self._tasks.keys():
            outputs[name] = inputs.pop(name)

        return outputs

    def call(self, logits: tf.Tensor, **kwargs):
        outputs = {}

        for name, task in self._tasks.items():
            predictions = self._tasks_prepares[name](logits, **kwargs) if name in self._tasks_prepares else logits

            outputs[name] = predictions

        return outputs

    def compute_loss(self, targets: Dict[Text, tf.Tensor], logits: Dict[Text, tf.Tensor], **kwargs) -> tf.Tensor:
        losses = []

        for name, task in self._tasks.items():
            target, predictions = targets[name], logits[name]
            # predictions = self._tasks_prepares[name](logits) if name in self._tasks_prepares else logits
            # print(predictions)

            losses.append(self._tasks[name](target, predictions, **kwargs) * self._task_weights[name])

        return tf.reduce_sum(losses)


class ModelWithHead(Model):
    def __init__(self, model: Model, head: Head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.head = head

    def call(self, inputs, **kwargs):
        return self.head(self.model(inputs, **kwargs), **kwargs)

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        targets = self.head.pop_labels(inputs)
        logits = self(inputs, training=training)

        return self.head.compute_loss(targets, logits)

    def get_config(self):
        pass
