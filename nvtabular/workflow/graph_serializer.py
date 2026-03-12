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
"""
JSON + artifact-based workflow serialization.

Saves a Workflow as:
  saved_workflow/
    metadata.json         (version info, written by Workflow.save)
    graph.json            (DAG topology + operator configs + inline fitted state)
    artifacts/
      node_<id>/          (one subdir per operator with file-based fitted state)
        ...parquet files...

No pickle is used.  Lambda functions cannot be serialized; named functions
defined in importable modules are supported.
"""

import importlib
import json
import os
import warnings

import merlin.dtypes as md
from merlin.dag.node import Node, iter_nodes
from merlin.dag.ops.concat_columns import ConcatColumns
from merlin.dag.ops.selection import SelectionOp
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema, ColumnSchema
from merlin.schema.tags import Tags, TagSet


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class WorkflowSerializationError(Exception):
    """Raised when a workflow cannot be serialized to the JSON format."""


# ---------------------------------------------------------------------------
# Callable serialization
# ---------------------------------------------------------------------------

_LAMBDA_WARNING = (
    "Lambda functions cannot be serialized by the JSON workflow serializer.\n"
    "Replace the lambda with a named function defined in an importable module:\n\n"
    "  # Instead of:\n"
    "  LambdaOp(lambda x: x * 2)\n\n"
    "  # Do:\n"
    "  def double(x):\n"
    "      return x * 2\n"
    "  LambdaOp(double)\n\n"
    "The function must be importable on the system where the workflow will be\n"
    "loaded (i.e., not defined in __main__ or a Jupyter notebook cell)."
)


def _callable_to_dict(f):
    """Serialize a callable to a JSON-safe dict.

    Raises WorkflowSerializationError for lambdas or __main__-defined functions.
    Returns None when f is None.
    """
    if f is None:
        return None
    if f.__name__ == "<lambda>":
        raise WorkflowSerializationError(
            f"Cannot serialize {f!r}.\n\n{_LAMBDA_WARNING}"
        )
    if f.__module__ == "__main__":
        raise WorkflowSerializationError(
            f"Cannot serialize '{f.__qualname__}': it is defined in __main__. "
            "Move it to an importable module."
        )
    return {"module": f.__module__, "qualname": f.__qualname__}


def _callable_from_dict(d):
    """Reconstruct a callable from its serialized dict. Returns None when d is None."""
    if d is None:
        return None
    mod = importlib.import_module(d["module"])
    obj = mod
    for part in d["qualname"].split("."):
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Tag serialization
# ---------------------------------------------------------------------------

def _tags_to_list(tags):
    """Serialize tags to a JSON-safe list of strings."""
    if not tags:
        return []
    return [str(t) for t in tags]


def _tags_from_list(tag_strs):
    """Reconstruct a list of Tags from serialized strings."""
    result = []
    for s in tag_strs:
        # Tags stringify as "Tags.CATEGORICAL"; strip the prefix
        name = s.split(".")[-1]
        try:
            result.append(Tags[name])
        except KeyError:
            result.append(s)  # keep as raw string; merlin accepts strings too
    return result


# ---------------------------------------------------------------------------
# DType serialization
# ---------------------------------------------------------------------------

def _dtype_to_dict(dtype):
    """Convert a merlin DType (or numpy dtype) to a JSON-safe dict preserving all fields."""
    if dtype is None:
        return None
    if isinstance(dtype, str):
        return {"name": dtype}
    # numpy dtype (has .str but not .element_type)
    if not hasattr(dtype, "element_type"):
        return {"name": str(dtype)}
    # merlin DType dataclass
    d = {"name": dtype.name}
    if dtype.element_type is not None:
        d["element_type"] = dtype.element_type.value
    if dtype.element_size is not None:
        d["element_size"] = dtype.element_size
    if dtype.element_unit is not None:
        d["element_unit"] = dtype.element_unit.value
    if dtype.signed is not None:
        d["signed"] = dtype.signed
    if dtype.shape is not None and dtype.shape.dims is not None:
        d["shape"] = [
            {"min": dim.min, "max": dim.max} for dim in dtype.shape.dims
        ]
    return d


def _dtype_from_dict(d):
    """Reconstruct a merlin DType from a serialized dict (or legacy string)."""
    if d is None:
        return None
    # Legacy support: plain string
    if isinstance(d, str):
        try:
            return md.dtype(d)
        except Exception:
            return None
    try:
        from merlin.dtypes.base import ElementType, ElementUnit
        from merlin.dtypes.shape import Shape, Dimension

        element_type = None
        if "element_type" in d:
            element_type = ElementType(d["element_type"])
        element_unit = None
        if "element_unit" in d:
            element_unit = ElementUnit(d["element_unit"])
        shape = None
        if "shape" in d:
            dims = tuple(Dimension(dim.get("min"), dim.get("max")) for dim in d["shape"])
            shape = Shape(dims)
        return md.DType(
            name=d["name"],
            element_type=element_type,
            element_size=d.get("element_size"),
            element_unit=element_unit,
            signed=d.get("signed"),
            shape=shape,
        )
    except Exception:
        # Fallback: try simple string-based reconstruction
        try:
            return md.dtype(d.get("name", ""))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Schema serialization
# ---------------------------------------------------------------------------

def _column_schema_to_dict(cs):
    return {
        "name": cs.name,
        "tags": _tags_to_list(cs.tags),
        "properties": cs.properties,
        "dtype": _dtype_to_dict(cs.dtype),
        "is_list": cs.is_list,
        "is_ragged": cs.is_ragged,
    }


def _column_schema_from_dict(d):
    return ColumnSchema(
        name=d["name"],
        tags=_tags_from_list(d.get("tags", [])),
        properties=d.get("properties", {}),
        dtype=_dtype_from_dict(d.get("dtype")),
        is_list=d.get("is_list"),
        is_ragged=d.get("is_ragged"),
    )


def _schema_to_dict(schema):
    if schema is None:
        return None
    return [_column_schema_to_dict(cs) for cs in schema.column_schemas.values()]


def _schema_from_dict(d):
    if d is None:
        return None
    return Schema([_column_schema_from_dict(item) for item in d])


# ---------------------------------------------------------------------------
# ColumnSelector serialization
# ---------------------------------------------------------------------------

def _selector_to_dict(selector):
    if selector is None:
        return None
    return {
        "names": list(selector.names),
        "tags": _tags_to_list(selector.tags) if hasattr(selector, "tags") else [],
    }


def _selector_from_dict(d):
    if d is None:
        return None
    return ColumnSelector(d["names"])


# ---------------------------------------------------------------------------
# Artifact helpers (file-based fitted state for StatOperators)
# ---------------------------------------------------------------------------

def _categories_to_json(categories_dict, artifact_dir, fs):
    """
    Copy artifact files/dirs to artifact_dir using set_storage_path has already been
    called. Serialize the categories dict as a list of {"key": ..., "path": ...} records
    where path is relative to artifact_dir.
    """
    if not categories_dict:
        return []
    records = []
    for k, v in categories_dict.items():
        key = list(k) if isinstance(k, tuple) else [k]
        # v is an absolute path; store the path relative to artifact_dir
        # (preserves subdirectory structure, e.g. "categories/unique.col.parquet")
        path_str = str(v)
        rel = os.path.relpath(path_str, artifact_dir)
        records.append({"key": key, "path": rel})
    return records


def _categories_from_json(records, artifact_dir):
    """Reconstruct a categories dict from serialized records + absolute artifact_dir."""
    if not records:
        return {}
    result = {}
    for item in records:
        key_list = item["key"]
        k = tuple(key_list) if len(key_list) > 1 else key_list[0]
        result[k] = os.path.join(artifact_dir, item["path"])
    return result


# ---------------------------------------------------------------------------
# Per-operator serialization functions
# (to_dict, from_dict) per class
# ---------------------------------------------------------------------------

# ---- merlin.dag ops ----

def _selection_op_to_dict(op, artifact_dir, fs):
    selector = op.selector if hasattr(op, "selector") else None
    return (
        {"selector": _selector_to_dict(selector)},
        {},
    )


def _selection_op_from_dict(params, state, artifact_dir):
    sel_dict = params.get("selector")
    selector = _selector_from_dict(sel_dict) if sel_dict else None
    return SelectionOp(selector)


def _concat_columns_to_dict(op, artifact_dir, fs):
    return ({}, {})


def _concat_columns_from_dict(params, state, artifact_dir):
    return ConcatColumns()


# ---- FillMissing ----

def _fill_missing_to_dict(op, artifact_dir, fs):
    return (
        {"fill_val": op.fill_val, "add_binary_cols": op.add_binary_cols},
        {},
    )


def _fill_missing_from_dict(params, state, artifact_dir):
    from nvtabular.ops.fill import FillMissing
    return FillMissing(
        fill_val=params.get("fill_val", 0),
        add_binary_cols=params.get("add_binary_cols", False),
    )


# ---- FillMedian ----

def _fill_median_to_dict(op, artifact_dir, fs):
    return (
        {"add_binary_cols": op.add_binary_cols},
        {"medians": {str(k): float(v) for k, v in op.medians.items()}},
    )


def _fill_median_from_dict(params, state, artifact_dir):
    from nvtabular.ops.fill import FillMedian
    op = FillMedian(add_binary_cols=params.get("add_binary_cols", False))
    op.medians = {k: float(v) for k, v in state.get("medians", {}).items()}
    return op


# ---- Normalize ----

def _normalize_to_dict(op, artifact_dir, fs):
    return (
        {"out_dtype": _dtype_to_dict(op.out_dtype) if op.out_dtype is not None else None},
        {
            "means": {str(k): float(v) for k, v in op.means.items()},
            "stds": {str(k): float(v) for k, v in op.stds.items()},
        },
    )


def _normalize_from_dict(params, state, artifact_dir):
    from nvtabular.ops.normalize import Normalize
    import numpy as np
    out_dtype_raw = params.get("out_dtype")
    out_dtype = np.dtype(out_dtype_raw["name"]) if out_dtype_raw else None
    op = Normalize(out_dtype=out_dtype)
    op.means = {k: float(v) for k, v in state.get("means", {}).items()}
    op.stds = {k: float(v) for k, v in state.get("stds", {}).items()}
    return op


# ---- NormalizeMinMax ----

def _normalize_minmax_to_dict(op, artifact_dir, fs):
    return (
        {"out_dtype": _dtype_to_dict(op.out_dtype) if op.out_dtype is not None else None},
        {
            "mins": {str(k): float(v) for k, v in op.mins.items()},
            "maxs": {str(k): float(v) for k, v in op.maxs.items()},
        },
    )


def _normalize_minmax_from_dict(params, state, artifact_dir):
    from nvtabular.ops.normalize import NormalizeMinMax
    import numpy as np
    out_dtype_raw = params.get("out_dtype")
    out_dtype = np.dtype(out_dtype_raw["name"]) if out_dtype_raw else None
    op = NormalizeMinMax(out_dtype=out_dtype)
    op.mins = {k: float(v) for k, v in state.get("mins", {}).items()}
    op.maxs = {k: float(v) for k, v in state.get("maxs", {}).items()}
    return op


# ---- Clip ----

def _clip_to_dict(op, artifact_dir, fs):
    return (
        {"min_value": op.min_value, "max_value": op.max_value},
        {},
    )


def _clip_from_dict(params, state, artifact_dir):
    from nvtabular.ops.clip import Clip
    return Clip(min_value=params.get("min_value"), max_value=params.get("max_value"))


# ---- LogOp ----

def _logop_to_dict(op, artifact_dir, fs):
    return ({}, {})


def _logop_from_dict(params, state, artifact_dir):
    from nvtabular.ops.logop import LogOp
    return LogOp()


# ---- HashBucket ----

def _hash_bucket_to_dict(op, artifact_dir, fs):
    return ({"num_buckets": op.num_buckets}, {})


def _hash_bucket_from_dict(params, state, artifact_dir):
    from nvtabular.ops.hash_bucket import HashBucket
    return HashBucket(num_buckets=params["num_buckets"])


# ---- Bucketize ----

def _bucketize_to_dict(op, artifact_dir, fs):
    # Requires _original_boundaries set in Bucketize.__init__
    boundaries = getattr(op, "_original_boundaries", None)
    if boundaries is None:
        raise WorkflowSerializationError(
            "Bucketize operator is missing _original_boundaries. "
            "Ensure nvtabular/ops/bucketize.py has been patched."
        )
    return ({"boundaries": boundaries}, {})


def _bucketize_from_dict(params, state, artifact_dir):
    from nvtabular.ops.bucketize import Bucketize
    return Bucketize(boundaries=params["boundaries"])


# ---- ListSlice ----

def _list_slice_to_dict(op, artifact_dir, fs):
    return (
        {
            "start": op.start,
            "end": op.end,
            "pad": op.pad,
            "pad_value": op.pad_value,
        },
        {},
    )


def _list_slice_from_dict(params, state, artifact_dir):
    from nvtabular.ops.list_slice import ListSlice
    return ListSlice(
        start=params["start"],
        end=params.get("end"),
        pad=params.get("pad", False),
        pad_value=params.get("pad_value", 0.0),
    )


# ---- Dropna ----

def _dropna_to_dict(op, artifact_dir, fs):
    return ({}, {})


def _dropna_from_dict(params, state, artifact_dir):
    from nvtabular.ops.dropna import Dropna
    return Dropna()


# ---- AddMetadata and subclasses ----

def _add_metadata_to_dict(op, artifact_dir, fs):
    return (
        {
            "tags": _tags_to_list(op.tags),
            "properties": op.properties,
        },
        {},
    )


def _add_metadata_from_dict(params, state, artifact_dir):
    from nvtabular.ops.add_metadata import AddMetadata
    return AddMetadata(
        tags=_tags_from_list(params.get("tags", [])),
        properties=params.get("properties", {}),
    )


# ---- Rename ----

def _rename_to_dict(op, artifact_dir, fs):
    f_dict = None
    if op.f is not None:
        f_dict = _callable_to_dict(op.f)  # raises WorkflowSerializationError for lambdas
    return (
        {"f": f_dict, "postfix": op.postfix, "name": op.name},
        {},
    )


def _rename_from_dict(params, state, artifact_dir):
    from nvtabular.ops.rename import Rename
    f = _callable_from_dict(params.get("f"))
    return Rename(f=f, postfix=params.get("postfix"), name=params.get("name"))


# ---- LambdaOp / UDF ----

def _lambdaop_to_dict(op, artifact_dir, fs):
    f = getattr(op, "f", None) or getattr(op, "func", None)
    f_dict = _callable_to_dict(f)  # raises WorkflowSerializationError for lambdas

    dep = getattr(op, "dependency", [])
    dep_names = []
    if dep:
        if isinstance(dep, str):
            dep_names = [dep]
        elif isinstance(dep, list):
            dep_names = [str(d) for d in dep]

    return (
        {
            "f": f_dict,
            "dependency": dep_names,
            "dtype": _dtype_to_dict(getattr(op, "dtype", None)),
            "tags": _tags_to_list(getattr(op, "tags", [])),
            "properties": getattr(op, "properties", {}),
        },
        {},
    )


def _lambdaop_from_dict(params, state, artifact_dir):
    from nvtabular.ops.lambdaop import LambdaOp
    f = _callable_from_dict(params.get("f"))
    dep = params.get("dependency") or []
    dtype_str = params.get("dtype")
    dtype = _dtype_from_dict(dtype_str)
    tags = _tags_from_list(params.get("tags", []))
    props = params.get("properties", {})
    return LambdaOp(
        f,
        dependency=dep if dep else None,
        dtype=dtype,
        tags=tags if tags else None,
        properties=props if props else None,
    )


# ---- Filter ----

def _filter_to_dict(op, artifact_dir, fs):
    return ({"f": _callable_to_dict(op.f)}, {})


def _filter_from_dict(params, state, artifact_dir):
    from nvtabular.ops.filter import Filter
    return Filter(f=_callable_from_dict(params["f"]))


# ---- Categorify ----

def _categorify_to_dict(op, artifact_dir, fs):
    # Copy all category artifact files to artifact_dir
    fs.makedirs(artifact_dir, exist_ok=True)
    op.set_storage_path(artifact_dir, copy=True)

    dtype_dict = None
    if op.dtype is not None:
        try:
            import numpy as np
            dtype_dict = {"name": np.dtype(op.dtype).str}
        except Exception:
            dtype_dict = {"name": str(op.dtype)}

    num_buckets = op.num_buckets
    max_size = op.max_size

    params = {
        "freq_threshold": op.freq_threshold,
        "cat_cache": op.cat_cache if isinstance(op.cat_cache, str) else "host",
        "dtype": dtype_dict,
        "on_host": op.on_host,
        "encode_type": op.encode_type,
        "name_sep": op.name_sep,
        "search_sorted": op.search_sorted,
        "num_buckets": num_buckets if isinstance(num_buckets, (int, type(None))) else dict(num_buckets),
        "max_size": max_size if isinstance(max_size, (int, type(None))) else dict(max_size),
        "single_table": op.single_table,
        "cardinality_memory_limit": str(op.cardinality_memory_limit)
        if op.cardinality_memory_limit is not None
        else None,
        "split_out": op.split_out if isinstance(op.split_out, (int, type(None))) else dict(op.split_out),
        "split_every": op.split_every if isinstance(op.split_every, (int, type(None))) else dict(op.split_every),
    }
    state = {
        "categories": _categories_to_json(op.categories, artifact_dir, fs),
        "storage_name": {str(k): str(v) for k, v in op.storage_name.items()},
    }
    return params, state


def _categorify_from_dict(params, state, artifact_dir):
    from nvtabular.ops.categorify import Categorify
    import numpy as np

    dtype_raw = params.get("dtype")
    dtype = np.dtype(dtype_raw["name"]) if dtype_raw else None

    op = Categorify(
        freq_threshold=params.get("freq_threshold", 0),
        cat_cache=params.get("cat_cache", "host"),
        dtype=dtype,
        on_host=params.get("on_host", True),
        encode_type=params.get("encode_type", "joint"),
        name_sep=params.get("name_sep", "_"),
        search_sorted=params.get("search_sorted", False),
        num_buckets=params.get("num_buckets"),
        max_size=params.get("max_size", 0),
        single_table=params.get("single_table", False),
        cardinality_memory_limit=params.get("cardinality_memory_limit"),
        split_out=params.get("split_out", 1),
        split_every=params.get("split_every", 8),
    )
    op.categories = _categories_from_json(state.get("categories", []), artifact_dir)
    op.out_path = artifact_dir
    op.storage_name = {k: v for k, v in state.get("storage_name", {}).items()}
    return op


# ---- TargetEncoding ----

def _target_encoding_to_dict(op, artifact_dir, fs):
    fs.makedirs(artifact_dir, exist_ok=True)
    op.set_storage_path(artifact_dir, copy=True)

    # Serialize the target column name(s)
    target = op.target
    if hasattr(target, "output_schema") and target.output_schema:
        target_cols = target.output_schema.column_names
    elif hasattr(target, "selector") and target.selector:
        target_cols = list(target.selector.names)
    else:
        target_cols = []

    out_dtype_dict = None
    if op.out_dtype is not None:
        try:
            import numpy as np
            out_dtype_dict = {"name": np.dtype(op.out_dtype).str}
        except Exception:
            out_dtype_dict = {"name": str(op.out_dtype)}

    params = {
        "target_cols": target_cols,
        "target_mean": op.target_mean,
        "kfold": op.kfold,
        "fold_seed": op.fold_seed,
        "p_smooth": op.p_smooth,
        "out_col": op.out_col,
        "out_dtype": out_dtype_dict,
        "split_out": op.split_out,
        "split_every": op.split_every,
        "on_host": op.on_host,
        "cat_cache": op.cat_cache if isinstance(op.cat_cache, str) else "host",
        "name_sep": op.name_sep,
        "drop_folds": op.drop_folds,
    }
    state = {
        "stats": _categories_to_json(op.stats, artifact_dir, fs),
        "means": {str(k): float(v) for k, v in op.means.items()},
    }
    return params, state


def _target_encoding_from_dict(params, state, artifact_dir):
    from nvtabular.ops.target_encoding import TargetEncoding
    import numpy as np

    target_cols = params.get("target_cols", [])
    out_dtype_raw = params.get("out_dtype")
    out_dtype = np.dtype(out_dtype_raw["name"]) if out_dtype_raw else None

    op = TargetEncoding(
        target=target_cols,
        target_mean=params.get("target_mean"),
        kfold=params.get("kfold", 3),
        fold_seed=params.get("fold_seed", 42),
        p_smooth=params.get("p_smooth", 20),
        out_col=params.get("out_col"),
        out_dtype=out_dtype,
        split_out=params.get("split_out"),
        split_every=params.get("split_every"),
        on_host=params.get("on_host", True),
        cat_cache=params.get("cat_cache", "host"),
        name_sep=params.get("name_sep", "_"),
        drop_folds=params.get("drop_folds", True),
    )
    op.stats = _categories_from_json(state.get("stats", []), artifact_dir)
    op.means = {k: float(v) for k, v in state.get("means", {}).items()}
    op.out_path = artifact_dir
    return op


# ---- Subgraph ----

def _subgraph_to_dict(op, artifact_dir, fs):
    loop_until_dict = None
    if op.loop_until is not None:
        loop_until_dict = _callable_to_dict(op.loop_until)

    # Recursively serialize the inner Graph.
    # Pass artifact_dir as the "path" so inner node artifacts land at
    # artifact_dir/artifacts/node_<inner_id>/ (separate from outer nodes).
    inner_nodes = _collect_all_nodes(op.graph.output_node)
    inner_node_to_id, _ = _build_id_maps(inner_nodes)
    inner_output_node_id = inner_node_to_id[id(op.graph.output_node)]

    inner_nodes_json = []
    for inner_node in inner_nodes:
        inner_node_id = inner_node_to_id[id(inner_node)]
        record = _serialize_node(inner_node, inner_node_id, inner_node_to_id, artifact_dir, fs)
        inner_nodes_json.append(record)

    params = {
        "name": op.name,
        "loop_until": loop_until_dict,
        "inner_graph": {
            "output_node_id": inner_output_node_id,
            "nodes": inner_nodes_json,
        },
    }
    return params, {}


def _subgraph_from_dict(params, state, artifact_dir):
    from merlin.dag.ops.subgraph import Subgraph
    from merlin.dag.graph import Graph

    loop_until = _callable_from_dict(params.get("loop_until"))

    inner_graph_data = params["inner_graph"]
    records = sorted(inner_graph_data["nodes"], key=lambda r: r["id"])
    node_map = {}
    for record in records:
        # artifact_dir acts as "path" for inner nodes; their artifact dirs become
        # artifact_dir/artifacts/node_<inner_id>/ (matching what was saved)
        node = _deserialize_node(record, node_map, artifact_dir, None)
        node_map[record["id"]] = node

    inner_output_node = node_map[inner_graph_data["output_node_id"]]
    inner_graph = Graph(inner_output_node)

    return Subgraph(name=params["name"], output_node=inner_graph, loop_until=loop_until)


# ---- JoinGroupby ----

def _join_groupby_to_dict(op, artifact_dir, fs):
    fs.makedirs(artifact_dir, exist_ok=True)
    op.set_storage_path(artifact_dir, copy=True)

    # cont_cols: serialize the column names if available
    cont_col_names = None
    if op._cont_names is not None:
        cont_col_names = list(op._cont_names.names)

    params = {
        "cont_cols": cont_col_names,
        "stats": list(op.stats),
        "split_out": op.split_out,
        "split_every": op.split_every,
        "on_host": op.on_host,
        "cat_cache": op.cat_cache if isinstance(op.cat_cache, str) else "host",
        "name_sep": op.name_sep,
    }
    state = {
        "categories": _categories_to_json(op.categories, artifact_dir, fs),
        "storage_name": {str(k): str(v) for k, v in op.storage_name.items()},
    }
    return params, state


def _join_groupby_from_dict(params, state, artifact_dir):
    from nvtabular.ops.join_groupby import JoinGroupby
    cont_cols = params.get("cont_cols")
    op = JoinGroupby(
        cont_cols=cont_cols,
        stats=tuple(params.get("stats", ("count",))),
        split_out=params.get("split_out"),
        split_every=params.get("split_every"),
        on_host=params.get("on_host", True),
        cat_cache=params.get("cat_cache", "host"),
        name_sep=params.get("name_sep", "_"),
    )
    op.categories = _categories_from_json(state.get("categories", []), artifact_dir)
    op.out_path = artifact_dir
    op.storage_name = {k: v for k, v in state.get("storage_name", {}).items()}
    return op


# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------
# Maps fully-qualified class name -> (class, to_dict_fn, from_dict_fn)

_REGISTRY = {}


def _register(class_path, cls, to_dict_fn, from_dict_fn):
    _REGISTRY[class_path] = (cls, to_dict_fn, from_dict_fn)


def _build_registry():
    from nvtabular.ops.fill import FillMissing, FillMedian
    from nvtabular.ops.normalize import Normalize, NormalizeMinMax
    from nvtabular.ops.clip import Clip
    from nvtabular.ops.logop import LogOp
    from nvtabular.ops.hash_bucket import HashBucket
    from nvtabular.ops.bucketize import Bucketize
    from nvtabular.ops.list_slice import ListSlice
    from nvtabular.ops.dropna import Dropna
    from nvtabular.ops.add_metadata import AddMetadata
    from nvtabular.ops.rename import Rename
    from nvtabular.ops.lambdaop import LambdaOp
    from nvtabular.ops.filter import Filter
    from nvtabular.ops.categorify import Categorify
    from nvtabular.ops.target_encoding import TargetEncoding
    from nvtabular.ops.join_groupby import JoinGroupby

    entries = [
        ("merlin.dag.ops.selection.SelectionOp", SelectionOp, _selection_op_to_dict, _selection_op_from_dict),
        ("merlin.dag.ops.concat_columns.ConcatColumns", ConcatColumns, _concat_columns_to_dict, _concat_columns_from_dict),
        ("nvtabular.ops.fill.FillMissing", FillMissing, _fill_missing_to_dict, _fill_missing_from_dict),
        ("nvtabular.ops.fill.FillMedian", FillMedian, _fill_median_to_dict, _fill_median_from_dict),
        ("nvtabular.ops.normalize.Normalize", Normalize, _normalize_to_dict, _normalize_from_dict),
        ("nvtabular.ops.normalize.NormalizeMinMax", NormalizeMinMax, _normalize_minmax_to_dict, _normalize_minmax_from_dict),
        ("nvtabular.ops.clip.Clip", Clip, _clip_to_dict, _clip_from_dict),
        ("nvtabular.ops.logop.LogOp", LogOp, _logop_to_dict, _logop_from_dict),
        ("nvtabular.ops.hash_bucket.HashBucket", HashBucket, _hash_bucket_to_dict, _hash_bucket_from_dict),
        ("nvtabular.ops.bucketize.Bucketize", Bucketize, _bucketize_to_dict, _bucketize_from_dict),
        ("nvtabular.ops.list_slice.ListSlice", ListSlice, _list_slice_to_dict, _list_slice_from_dict),
        ("nvtabular.ops.dropna.Dropna", Dropna, _dropna_to_dict, _dropna_from_dict),
        ("nvtabular.ops.add_metadata.AddMetadata", AddMetadata, _add_metadata_to_dict, _add_metadata_from_dict),
        ("nvtabular.ops.rename.Rename", Rename, _rename_to_dict, _rename_from_dict),
        ("nvtabular.ops.lambdaop.LambdaOp", LambdaOp, _lambdaop_to_dict, _lambdaop_from_dict),
        ("nvtabular.ops.filter.Filter", Filter, _filter_to_dict, _filter_from_dict),
        ("nvtabular.ops.categorify.Categorify", Categorify, _categorify_to_dict, _categorify_from_dict),
        ("nvtabular.ops.target_encoding.TargetEncoding", TargetEncoding, _target_encoding_to_dict, _target_encoding_from_dict),
        ("nvtabular.ops.join_groupby.JoinGroupby", JoinGroupby, _join_groupby_to_dict, _join_groupby_from_dict),
    ]
    for class_path, cls, to_dict_fn, from_dict_fn in entries:
        _register(class_path, cls, to_dict_fn, from_dict_fn)

    # Also register UDF under its merlin.dag path (LambdaOp is an alias for UDF)
    try:
        from merlin.dag.ops.udf import UDF
        _register("merlin.dag.ops.udf.UDF", UDF, _lambdaop_to_dict, _lambdaop_from_dict)
    except ImportError:
        pass

    # Register Subgraph (from merlin-core); falls back to class-path-only lookup
    try:
        from merlin.dag.ops.subgraph import Subgraph as _Subgraph
        _register(
            "merlin.dag.ops.subgraph.Subgraph",
            _Subgraph,
            _subgraph_to_dict,
            _subgraph_from_dict,
        )
    except ImportError:
        # Still register by class-path string so the serializer works if Subgraph
        # is somehow instantiated via another import path
        _register(
            "merlin.dag.ops.subgraph.Subgraph",
            None,
            _subgraph_to_dict,
            _subgraph_from_dict,
        )


def _get_op_class_path(op):
    """Return the fully-qualified class path of an operator."""
    cls = type(op)
    return f"{cls.__module__}.{cls.__qualname__}"


def _lookup_serializer(op):
    """Return (class_path, to_dict_fn, from_dict_fn) for the given operator."""
    if not _REGISTRY:
        _build_registry()

    class_path = _get_op_class_path(op)
    entry = _REGISTRY.get(class_path)
    if entry is not None:
        return class_path, entry[1], entry[2]

    # Fall back: check if the operator's class is a subclass of a registered class
    for reg_path, (reg_cls, to_dict_fn, from_dict_fn) in _REGISTRY.items():
        if reg_cls is not None and isinstance(op, reg_cls):
            return reg_path, to_dict_fn, from_dict_fn

    _DEFERRED = {
        "DifferenceLag", "ValueCount", "DataStats", "DropLowCardinality",
        "ColumnSimilarity", "JoinExternal", "HashedCross", "Groupby",
        "ReduceDtypeSize", "SubsetColumns", "SubtractionOp", "GroupingOp",
    }
    class_name = type(op).__name__
    if class_name in _DEFERRED:
        raise NotImplementedError(
            f"The operator '{class_name}' is not yet supported by the JSON workflow "
            "serializer. Please open an issue or use a supported operator."
        )

    raise WorkflowSerializationError(
        f"No serializer registered for operator '{class_path}'. "
        "Register one via nvtabular.workflow.graph_serializer._register() "
        "or open an issue."
    )


# ---------------------------------------------------------------------------
# Graph traversal helpers
# ---------------------------------------------------------------------------

def _collect_all_nodes(output_node):
    """
    BFS from output_node backwards through parents and dependencies.
    Returns nodes in topological order: leaves (no parents) first, output node last.
    """
    visited_ids = set()
    bfs_order = []
    queue = [output_node]

    while queue:
        node = queue.pop(0)
        nid = id(node)
        if nid in visited_ids:
            continue
        visited_ids.add(nid)
        bfs_order.append(node)
        for connected in node.parents_with_dependencies:
            if id(connected) not in visited_ids:
                queue.append(connected)

    # Reverse: leaves come first, output node is last
    bfs_order.reverse()
    return bfs_order


def _build_id_maps(nodes):
    """
    Returns:
      node_to_id: dict mapping id(node) -> integer id
      id_to_node: dict mapping integer id -> node object
    """
    node_to_id = {}
    id_to_node = {}
    for i, node in enumerate(nodes):
        node_to_id[id(node)] = i
        id_to_node[i] = node
    return node_to_id, id_to_node


# ---------------------------------------------------------------------------
# Node serialization / deserialization
# ---------------------------------------------------------------------------

def _serialize_node(node, node_integer_id, node_to_id, path, fs):
    artifact_dir = fs.sep.join([path, "artifacts", f"node_{node_integer_id}"])

    op = node.op
    if op is None:
        op_class_path = None
        op_params = {}
        op_state = {}
    else:
        try:
            op_class_path, to_dict_fn, _ = _lookup_serializer(op)
            op_params, op_state = to_dict_fn(op, artifact_dir, fs)
        except WorkflowSerializationError:
            raise
        except NotImplementedError:
            raise

    parent_ids = [node_to_id[id(p)] for p in node.parents]

    dep_ids = []
    for dep in node.dependencies:
        if isinstance(dep, list):
            dep_ids.extend(node_to_id[id(d)] for d in dep)
        else:
            dep_ids.append(node_to_id[id(dep)])

    return {
        "id": node_integer_id,
        "op_class": op_class_path,
        "op_params": op_params,
        "op_state": op_state,
        "parent_ids": parent_ids,
        "dependency_ids": dep_ids,
        "selector": _selector_to_dict(node.selector),
        "input_schema": _schema_to_dict(node.input_schema),
        "output_schema": _schema_to_dict(node.output_schema),
    }


def _deserialize_node(record, node_map, path, fs):
    """Reconstruct a Node from its record dict. node_map must already contain all parents."""
    node_integer_id = record["id"]
    artifact_dir = os.path.join(path, "artifacts", f"node_{node_integer_id}")

    op_class_path = record.get("op_class")
    op_params = record.get("op_params", {})
    op_state = record.get("op_state", {})

    if op_class_path is None:
        op = None
    else:
        if not _REGISTRY:
            _build_registry()
        entry = _REGISTRY.get(op_class_path)
        if entry is None:
            raise WorkflowSerializationError(
                f"Unknown operator class '{op_class_path}' in graph.json. "
                "Cannot deserialize this workflow."
            )
        _, from_dict_fn = entry[0], entry[2]
        op = from_dict_fn(op_params, op_state, artifact_dir)

        # Mark StatOperators as fitted — the fitted state has been restored from
        # the serialized op_state / artifact files above.
        if hasattr(op, "fitted"):
            op.fitted = True

    node = Node()
    node.op = op

    sel_dict = record.get("selector")
    if sel_dict:
        node.selector = _selector_from_dict(sel_dict)

    in_schema = record.get("input_schema")
    out_schema = record.get("output_schema")
    node.input_schema = _schema_from_dict(in_schema)
    node.output_schema = _schema_from_dict(out_schema)

    for pid in record.get("parent_ids", []):
        node.add_parent(node_map[pid])

    for did in record.get("dependency_ids", []):
        node.add_dependency(node_map[did])

    return node


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def serialize_graph(workflow, path, fs):
    """
    Serialize the workflow graph to ``path/graph.json`` and copy all
    file-based operator artifacts to ``path/artifacts/node_<id>/``.

    Parameters
    ----------
    workflow : Workflow
        The fitted workflow to serialize.
    path : str
        Root directory of the saved workflow (already created by caller).
    fs : fsspec filesystem
        Filesystem object to use for I/O.
    """
    if not _REGISTRY:
        _build_registry()

    nodes = _collect_all_nodes(workflow.output_node)
    node_to_id, _ = _build_id_maps(nodes)

    output_node_id = node_to_id[id(workflow.output_node)]

    nodes_json = []
    for node in nodes:
        node_integer_id = node_to_id[id(node)]
        record = _serialize_node(node, node_integer_id, node_to_id, path, fs)
        nodes_json.append(record)

    graph_data = {
        "format_version": 1,
        "output_node_id": output_node_id,
        "nodes": nodes_json,
    }

    graph_path = fs.sep.join([path, "graph.json"])
    with fs.open(graph_path, "w") as f:
        json.dump(graph_data, f, indent=2)


def deserialize_graph(path, fs):
    """
    Reconstruct a Workflow from ``path/graph.json`` and artifacts in
    ``path/artifacts/``.

    Parameters
    ----------
    path : str
        Root directory of the saved workflow.
    fs : fsspec filesystem
        Filesystem object to use for I/O.

    Returns
    -------
    Workflow
        The reconstructed workflow (without a Dask client; caller sets it).
    """
    if not _REGISTRY:
        _build_registry()

    graph_path = fs.sep.join([path, "graph.json"])
    with fs.open(graph_path, "r") as f:
        graph_data = json.load(f)

    format_version = graph_data.get("format_version", 1)
    if format_version != 1:
        raise WorkflowSerializationError(
            f"Unsupported graph.json format_version={format_version}. "
            "This version of nvtabular only supports format_version=1."
        )

    # Nodes are stored in topological order (parents before children).
    # Process them in id order to guarantee parents are ready before children.
    records = sorted(graph_data["nodes"], key=lambda r: r["id"])
    node_map = {}  # integer id -> Node
    for record in records:
        node = _deserialize_node(record, node_map, path, fs)
        node_map[record["id"]] = node

    output_node = node_map[graph_data["output_node_id"]]

    # Build the Workflow without going through __init__ (avoids re-creating the graph).
    from nvtabular.workflow.workflow import Workflow
    from merlin.dag import Graph
    from merlin.dag.executors import DaskExecutor

    workflow = object.__new__(Workflow)
    workflow.graph = Graph(output_node)
    workflow.executor = DaskExecutor(None)
    return workflow
