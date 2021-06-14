"""This code is taken from: https://github.com/tensorflow/recommenders"""

from typing import List, Optional, Text

import tensorflow as tf


class Model(tf.keras.Model):
    """Base model for TFRS models.
    Many recommender models are relatively complex, and do not neatly fit into
    supervised or unsupervised paradigms. This base class makes it easy to
    define custom training and test losses for such complex models.
    This is done by asking the user to implement the following methods:
    - `__init__` to set up your model. Variable, task, loss, and metric
      initialization should go here.
    - `compute_loss` to define the training loss. The method takes as input the
      raw features passed into the model, and returns a loss tensor for training.
      As part of doing so, it should also update the model's metrics.
    - [Optional] `call` to define how the model computes its predictions. This
      is not always necessary: for example, two-tower retrieval models have two
      well-defined submodels whose `call` methods are normally used directly.
    Note that this base class is a thin conveniece wrapper for tf.keras.Model, and
    equivalent functionality can easily be achieved by overriding the `train_step`
    and `test_step` methods of a plain Keras model. Doing so also makes it easy
    to build even more complex training mechanisms, such as the use of
    different optimizers for different variables, or manipulating gradients.
    Keras has an excellent tutorial on how to
    do this [here](
    https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
    """

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        """Defines the loss function.
        Args:
          inputs: A data structure of tensors: raw inputs to the model. These will
            usually contain labels and weights as well as features.
          training: Whether the model is in training mode.
        Returns:
          Loss tensor.
        """

        raise NotImplementedError(
            "Implementers must implement the `compute_loss` method.")

    def train_step(self, inputs):
        """Custom train step using the `compute_loss` method."""

        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs, training=True)

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, inputs):
        """Custom test step using the `compute_loss` method."""

        loss = self.compute_loss(inputs, training=False)

        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics


class Task:
    """Task class.
    This is a marker class: inherit from this class if you'd like to make
    your tasks distinguishable from plain Keras layers.
    """

    pass


class Ranking(tf.keras.layers.Layer, Task):
    """A ranking task.
    Recommender systems are often composed of two components:
    - a retrieval model, retrieving O(thousands) candidates from a corpus of
      O(millions) candidates.
    - a ranker model, scoring the candidates retrieved by the retrieval model to
      return a ranked shortlist of a few dozen candidates.
    This task helps with building ranker models. Usually, these will involve
    predicting signals such as clicks, cart additions, likes, ratings, and
    purchases.
    """

    def __init__(
            self,
            loss: Optional[tf.keras.losses.Loss] = None,
            metrics: Optional[List[tf.keras.metrics.Metric]] = None,
            prediction_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
            label_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
            loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
            name: Optional[Text] = None) -> None:
        """Initializes the task.
        Args:
          loss: Loss function. Defaults to BinaryCrossentropy.
          metrics: List of Keras metrics to be evaluated.
          prediction_metrics: List of Keras metrics used to summarize the
            predictions.
          label_metrics: List of Keras metrics used to summarize the labels.
          loss_metrics: List of Keras metrics used to summarize the loss.
          name: Optional task name.
        """

        super().__init__(name=name)

        self._loss = (
            loss if loss is not None else tf.keras.losses.BinaryCrossentropy())
        self._ranking_metrics = metrics or []
        self._prediction_metrics = prediction_metrics or []
        self._label_metrics = label_metrics or []
        self._loss_metrics = loss_metrics or []

    def call(self,
             labels: tf.Tensor,
             predictions: tf.Tensor,
             sample_weight: Optional[tf.Tensor] = None,
             training: bool = False,
             compute_metrics: bool = True) -> tf.Tensor:
        """Computes the task loss and metrics.
        Args:
          labels: Tensor of labels.
          predictions: Tensor of predictions.
          sample_weight: Tensor of sample weights.
          training: Indicator whether training or test loss is being computed.
          compute_metrics: Whether to compute metrics. Set this to False
            during training for faster training.
        Returns:
          loss: Tensor of loss values.
        """

        loss = self._loss(
            y_true=labels, y_pred=predictions, sample_weight=sample_weight)

        if not compute_metrics:
            return loss

        update_ops = []

        for metric in self._ranking_metrics:
            update_ops.append(metric.update_state(
                y_true=labels, y_pred=predictions, sample_weight=sample_weight))

        for metric in self._prediction_metrics:
            update_ops.append(
                metric.update_state(predictions, sample_weight=sample_weight))

        for metric in self._label_metrics:
            update_ops.append(
                metric.update_state(labels, sample_weight=sample_weight))

        for metric in self._loss_metrics:
            update_ops.append(
                metric.update_state(loss, sample_weight=sample_weight))

        # Custom metrics may not return update ops, unlike built-in
        # Keras metrics.
        update_ops = [x for x in update_ops if x is not None]

        with tf.control_dependencies(update_ops):
            return tf.identity(loss)
