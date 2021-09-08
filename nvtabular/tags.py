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
    ID_ITEM = "item_id"
    CONTEXT = "context"

    BINARY = "binary"
    REGRESSION = "regression"
    MULTI_CLASS = "multi_class"

    # Target related
    TARGETS = "target"
