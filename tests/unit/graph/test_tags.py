#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
import pytest

from nvtabular.graph.tags import Tags, TagSet


def test_tagset_init_normalizes_tags_to_enum():
    origin_tags = ["continuous", "list", "custom_tag"]
    tag_set = TagSet(origin_tags)
    assert Tags.CONTINUOUS in tag_set._tags
    assert Tags.LIST in tag_set._tags


def test_tagset_init_collision_error():
    with pytest.raises(ValueError) as err:
        tag_set = TagSet(["continuous", "categorical"])  # noqa

    assert "continuous" in str(err.value)
    assert "categorical" in str(err.value)
    assert "incompatible" in str(err.value)


def test_tagset_is_iterable():
    origin_tags = ["continuous", "list"]
    tag_set = TagSet(origin_tags)
    for tag in tag_set:
        assert tag.value in origin_tags
    assert len(tag_set) == len(origin_tags)


def test_tagset_add():
    origin_tags = [Tags.CONTINUOUS, Tags.LIST, "custom_tag"]
    tag_set = TagSet(origin_tags)

    new_tags = "custom_tag2"
    new_tag_set = tag_set + new_tags
    assert len(new_tag_set) == 4
    assert new_tags in new_tag_set
    assert all(origin_tag in new_tag_set for origin_tag in origin_tags)

    new_tags = ["custom_tag2"]
    new_tag_set = tag_set + new_tags

    assert len(new_tag_set) == 4
    assert all(new_tag in new_tag_set for new_tag in new_tags)
    assert all(origin_tag in new_tag_set for origin_tag in origin_tags)

    new_tags = TagSet(["custom_tag2"])
    new_tag_set = tag_set + new_tags

    assert len(new_tag_set) == 4
    assert all(tag in new_tag_set for tag in new_tags._tags)
    assert all(origin_tag in new_tag_set for origin_tag in origin_tags)


def test_tagset_sub():
    origin_tags = [Tags.CONTINUOUS, Tags.LIST]
    tag_set = TagSet(origin_tags + ["custom_tag"])
    assert len(tag_set) == 3

    new_tags = "custom_tag"
    new_tag_set = tag_set - new_tags
    assert len(new_tag_set) == 2
    assert new_tags not in new_tag_set
    assert all(origin_tag in new_tag_set for origin_tag in origin_tags)

    new_tags = ["custom_tag"]
    new_tag_set = tag_set - new_tags

    assert len(new_tag_set) == 2
    assert all(new_tag not in new_tag_set for new_tag in new_tags)
    assert all(origin_tag in new_tag_set for origin_tag in origin_tags)

    new_tags = TagSet(["custom_tag"])
    new_tag_set = tag_set - new_tags

    assert len(new_tag_set) == 2
    assert all(new_tag not in new_tag_set for new_tag in new_tags)
    assert all(origin_tag in new_tag_set for origin_tag in origin_tags)


def test_tagset_add_collision_error():
    origin_tags = ["continuous", "list", "custom_tag"]
    tag_set = TagSet(origin_tags)

    new_tags = "categorical"

    with pytest.raises(ValueError) as err:
        tag_set + new_tags  # pylint: disable=W0104

    assert "continuous" in str(err.value)
    assert "categorical" in str(err.value)
    assert "incompatible" in str(err.value)

    new_tags = ["categorical"]

    with pytest.raises(ValueError) as err:
        tag_set + new_tags  # pylint: disable=W0104

    assert "continuous" in str(err.value)
    assert "categorical" in str(err.value)
    assert "incompatible" in str(err.value)

    new_tags = TagSet(["categorical"])

    with pytest.raises(ValueError) as err:
        tag_set + new_tags  # pylint: disable=W0104

    assert "continuous" in str(err.value)
    assert "categorical" in str(err.value)
    assert "incompatible" in str(err.value)
