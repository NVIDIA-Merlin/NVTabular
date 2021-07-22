from dataclasses import dataclass, field
from typing import Dict, List, Optional, Text

import tree

from nvtabular.tag import DefaultTags


@dataclass(frozen=True)
class Column:
    """"A Column with metadata. """

    name: Text
    tags: Optional[List[Text]] = field(default_factory=list)
    properties: Optional[Dict[Text, Text]] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.tags, DefaultTags):
            object.__setattr__(self, "tags", self.tags.value)

    def __str__(self) -> str:
        return self.name

    def with_tags(self, tags, add=True) -> "Column":
        if isinstance(tags, DefaultTags):
            tags = tags.value
        if not tags:
            return self

        tags = list(set(list(self.tags) + list(tags))) if add else tags

        return Column(self.name, tags=tags, properties=self.properties)

    def with_name(self, name) -> "Column":
        return Column(name, tags=self.tags, properties=self.properties)

    def with_properties(self, add=True, **properties) -> "Column":
        if not properties:
            return self
        properties = {**self.properties, **properties} if add else properties

        return Column(self.name, tags=self.tags, properties=properties)


class Columns(list):
    def map(self, fn) -> "Columns":
        return tree.map_structure(fn, self)

    def map_names(self, fn) -> "Columns":
        return tree.map_structure(lambda x: x.with_name(fn(x.name)), self)

    def flatten(self) -> "Columns":
        return tree.flatten(self)
