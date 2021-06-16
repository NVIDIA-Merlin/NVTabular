from enum import Enum


class DefaultTags(Enum):
    # Feature types
    CATEGORICAL = ["categorical"]
    CONTINUOUS = ["continuous"]
    LIST = ["list"]
    IMAGE = ["image"]
    TEXT = ["text"]

    # Feature context
    USER = ["user"]
    ITEM = ["item"]
    CONTEXT = ["context"]

    # Target related
    TARGETS = ["target"]
    TARGETS_BINARY = ["target", "binary"]
    TARGETS_REGRESSION = ["target", "regression"]
    TARGETS_MULTI_CLASS = ["target", "multi_class"]


class Tag:
    CATEGORICAL = DefaultTags.CATEGORICAL
    CONTINUOUS = DefaultTags.CONTINUOUS
    LIST = DefaultTags.LIST
    IMAGE = DefaultTags.IMAGE
    TEXT = DefaultTags.TEXT

    # Feature context
    USER = DefaultTags.USER
    ITEM = DefaultTags.ITEM
    CONTEXT = DefaultTags.CONTEXT

    # Target related
    TARGETS = DefaultTags.TARGETS
    TARGETS_BINARY = DefaultTags.TARGETS_BINARY
    TARGETS_REGRESSION = DefaultTags.TARGETS_REGRESSION
    TARGETS_MULTI_CLASS = DefaultTags.TARGETS_MULTI_CLASS

    def __init__(self, *tag):
        self.tags = tag

    @classmethod
    def parse(cls, tag, allow_list=True):
        if allow_list and isinstance(tag, list):
            return Tag(tag)
        elif isinstance(tag, DefaultTags):
            return Tag(tag.value)
        elif isinstance(tag, Tag):
            return tag


class TagAs:
    def __init__(self, tags=None, is_target=False, is_regression_target=False, is_binary_target=False,
                 is_multi_class_target=False):
        if isinstance(tags, DefaultTags):
            tags = tags.value
        if not tags:
            tags = []
        if not isinstance(tags, list):
            tags = [tags]
        if is_target:
            tags.extend(DefaultTags.TARGETS.value)
        if is_regression_target:
            tags.extend(DefaultTags.TARGETS_REGRESSION.value)
        if is_multi_class_target:
            tags.extend(DefaultTags.TARGETS_MULTI_CLASS.value)
        if is_binary_target:
            tags.extend(DefaultTags.TARGETS_BINARY.value)

        self.tags = list(set(tags))
