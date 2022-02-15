#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
    TARGET = "target"
    BINARY = "binary"
    REGRESSION = "regression"
    MULTI_CLASS = "multi_class"


TAG_COLLISIONS = {
    Tags.CATEGORICAL: [Tags.CONTINUOUS],
    Tags.CONTINUOUS: [Tags.CATEGORICAL],
}


class TagSet:
    def __init__(self, tags=None):
        tags = tags or []
        if isinstance(tags, TagSet):
            tags = tags._tags

        self._tags = self._normalize_tags(tags)

        collisions = self._detect_collisions(self._tags, self._tags)
        if collisions:
            raise ValueError(
                f"Could not create a TagSet with the tags {self._tags}. "
                f"The following tags are incompatible: {collisions}"
            )

    def override(self, tags):
        tags = self._convert_to_tagset(tags)
        to_remove = self._detect_collisions(self._tags, tags)
        return TagSet(self - to_remove + tags)

    def __iter__(self):
        for tag in self._tags:
            yield tag

    def __len__(self):
        return len(self._tags)

    def __add__(self, tags):
        tags = self._convert_to_tagset(tags)
        return TagSet(self._tags.union(tags._tags))

    def __sub__(self, tags):
        tags = self._convert_to_tagset(tags)
        return TagSet(self._tags - tags._tags)

    def __eq__(self, tags):
        return self._tags == tags._tags

    def _detect_collisions(self, tags_a, tags_b):
        collisions = []
        for tag in tags_b:
            conflicting = TAG_COLLISIONS.get(tag, [])
            for conflict in conflicting:
                if conflict in tags_a:
                    collisions.append(conflict)
        return set(collisions)

    def _convert_to_tagset(self, tags):
        if not isinstance(tags, (list, set, TagSet)):
            tags = [tags]
        if not isinstance(tags, TagSet):
            tags = TagSet(tags)

        return tags

    def _normalize_tags(self, tags):
        return set(Tags[tag.upper()] if tag in Tags._value2member_map_ else tag for tag in tags)

    def __repr__(self) -> str:
        return str(self._tags)
