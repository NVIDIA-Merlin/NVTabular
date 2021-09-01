from enum import Enum


class DefaultTags(Enum):
    # Feature types
    CATEGORICAL = ["categorical"]
    CONTINUOUS = ["continuous"]
    LIST = ["list"]
    IMAGE = ["image"]
    TEXT = ["text"]
    TEXT_TOKENIZED = ["text_tokenized"]
    TIME = ["time"]

    # Feature context
    USER = ["user"]
    ITEM = ["item"]
    ITEM_ID = ["item", "item_id"]
    CONTEXT = ["context"]

    # Target related
    TARGETS = ["target"]
    TARGETS_BINARY = ["target", "binary"]
    TARGETS_REGRESSION = ["target", "regression"]
    TARGETS_MULTI_CLASS = ["target", "multi_class"]
