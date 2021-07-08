from dataclasses import dataclass, field
from typing import Dict, List, Optional, Text, Union


@dataclass(frozen=True)
class Column:
    """"A Column with metadata. """

    name: Text
    tags: Optional[List[Text]] = field(default_factory=list)
    properties: Optional[Dict[Text, Text]] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.name

    def add_tags(self, tags):
        if not tags:
            return self

        return Column(self.name, tags=self.tags + tags, properties=self.properties)

    def add_properties(self, **properties):
        if not properties:
            return self

        return Column(self.name, tags=self.tags, properties={**self.properties, **properties})
