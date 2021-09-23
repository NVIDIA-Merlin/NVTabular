from enum import Enum


class Tags(Enum):
    # Feature types
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    LIST = "list"
    TEXT = "text"
    TEXT_TOKENIZED = "text_tokenized"
    TIME = "time"

    # Feature context
    USER = "user"
    ITEM = "item"
    ITEM_ID = "item_id"
    CONTEXT = "context"

    # Target related
    TARGETS = "target"
    BINARY = "binary"
    REGRESSION = "regression"
    MULTI_CLASS = "multi_class"
