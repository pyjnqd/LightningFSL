"""Interfaces for learning specifications."""

import collections
import enum


class Split(enum.Enum):
  """The possible data splits."""
  TRAIN = 0
  VALID = 1
  TEST = 2


class BatchSpecification(
    collections.namedtuple('BatchSpecification', 'split, batch_size')):
  """The specification of an episode.

    Args:
      split: the Split from which to pick data.
      batch_size: an int, the number of (image, label) pairs in the batch.
  """
  pass


class EpisodeSpecification(
    collections.namedtuple(
        'EpisodeSpecification',
        'split, num_classes, num_train_examples, num_test_examples')):
  """The specification of an episode.

    Args:
      split: A Split from which to pick data.
      num_classes: The number of classes in the episode, or None for variable.
      num_train_examples: The number of examples to use per class in the train
        phase, or None for variable.
      num_test_examples: the number of examples to use per class in the test
        phase, or None for variable.
  """
