#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
import collections
import logging
import time
import warnings

import dask
import dask_cudf
import yaml
from fsspec.core import get_fs_token_paths

from nvtabular.io.dask import _ddf_to_dataset
from nvtabular.io.dataset import Dataset, _set_dtypes
from nvtabular.io.shuffle import Shuffle, _check_shuffle_arg
from nvtabular.io.writer_factory import writer_factory
from nvtabular.ops import DFOperator, Operator, StatOperator, TransformOperator
from nvtabular.worker import clean_worker_cache

LOG = logging.getLogger("nvtabular")


class BaseWorkflow:

    """
    BaseWorkflow class organizes and runs all the feature engineering
    and pre-processing operators for your workflow.

    Parameters
    -----------
    cat_names : list of str
        Names of the categorical columns.
    cont_names : list of str
        Names of the continuous columns.
    label_name : list of str
        Names of the label column.
    config : object
    """

    def __init__(self, cat_names=None, cont_names=None, label_name=None, config=None, delim="_"):
        self.phases = []

        self.columns_ctx = {}
        self.columns_ctx["all"] = {}
        self.columns_ctx["continuous"] = {}
        self.columns_ctx["categorical"] = {}
        self.columns_ctx["label"] = {}
        self.columns_ctx["all"]["base"] = cont_names + cat_names + label_name
        self.columns_ctx["continuous"]["base"] = cont_names
        self.columns_ctx["categorical"]["base"] = cat_names
        self.columns_ctx["label"]["base"] = label_name

        self.stats = {}
        self.current_file_num = 0
        self.delim = delim
        self.timings = {"write_df": 0.0, "preproc_apply": 0.0}
        self.ops_in = []
        if config:
            self.load_config(config)
        else:
            # create blank config and for later fill in
            self.config = get_new_config()

        self.clear_stats()

    def _register_ops(self, operators):
        if not isinstance(operators, list):
            operators = [operators]
        # order matters!!!
        if "full" not in self.columns_ctx:
            self.columns_ctx["full"] = {}
            # full current will always be the most up to date version of columns,
            # based on operators added
            self.columns_ctx["full"]["base"] = (self.columns_ctx["all"]["base"],)
            current = (self.columns_ctx["all"]["base"],)
        # for all what does not exist in
        for op, col_focus, chain_on_ops, child, parent in operators:
            target_cols = op.get_columns(self.columns_ctx, col_focus, chain_on_ops)
            # grab the input target columns and the input extra columns
            extra_cols = []
            cur_extra_cols = [] if len(current) < 2 else current[1]
            full_list = current[0] + cur_extra_cols
            for col in full_list:
                if col not in target_cols:
                    extra_cols.append(col)
            current = self._create_full_col_ctx_entry(op, target_cols, extra_cols, parent=parent)
        self._reduce(self.columns_ctx["full"])

    def _reduce(self, full_dict):
        self._remove_dupes(full_dict)
        # this will guide phase placement
        self._analyze_placement(full_dict)

    def _analyze_placement(self, full_dict):
        # detect num collisions for each op_id to find correct placement.
        self.placement = {}
        for op_id, cols_ops in full_dict.items():
            if op_id in "base":
                continue
            in_tar_cols, _, _, _, _, _ = cols_ops
            in_tar_cols = in_tar_cols if in_tar_cols else []
            self.placement[op_id] = self._detect_num_col_collisions(in_tar_cols.copy(), op_id)

    def _remove_dupes(self, full_dict):
        remove_keys = []
        for op_id, cols_ops in full_dict.items():
            parent = None
            if op_id not in "base":
                in_tar_cols, in_extra_cols, op, parent, fin_tar_cols, fin_extra_cols = cols_ops
            if not parent:
                continue
            if parent not in full_dict:
                remove_keys.append(op_id)
        for key in remove_keys:
            del full_dict[key]

    def _detect_cols_collision(self, columns, op_id, index=0):
        """
        Given a list of columns find the task AFTER which all
        columns in list exists and return that task.
        """
        action_cols = columns
        # start at the index provided, scan forward.
        keys = list(self.columns_ctx["full"].keys())
        keys = keys[index:]
        for idx, k in enumerate(keys):
            if k == op_id:
                # reach yourself is end, and last found location
                return k, index
            if k in "base":
                fin_tar_cols = self.columns_ctx["full"][k][0]
            else:
                (
                    in_tar_cols,
                    in_extra_cols,
                    op,
                    parent,
                    fin_tar_cols,
                    fin_extra_cols,
                ) = self.columns_ctx["full"][k]
            # if anything is inside, find it and remove it
            # found col, cannot remove in mid iteration... causes issues
            found = set(fin_tar_cols).intersection(action_cols)
            action_cols = [col for col in action_cols if col not in found]
            # if empty found end
            if not action_cols:
                return k, idx + index
        raise ValueError(f"Unknown columns found: {action_cols}")

    def _detect_num_col_collisions(self, columns, op_id):
        """
        Detect the number of times you see all columns in tasks, before getting
        to self in task list
        """
        current_op = None
        index = 0
        indexes = []
        while current_op != op_id:
            current_op, index = self._detect_cols_collision(columns.copy(), op_id, index=index)
            indexes.append((current_op, index))
            index = index + 1
        return indexes

    def _check_op_count(self, op):
        if op._id_set is None:
            count = self._get_op_count(op._id)
            # reset id based on count of op in workflow already
            op_id = f"{op._id}{self.delim}{str(count + 1)}"
            op._set_id(op_id)

    def _create_full_col_ctx_entry(self, op, target_cols, extra_cols, parent=None):
        if isinstance(parent, Operator):
            parent = parent._id
        tup_rep = None
        # requires target columns, extra columns (target+extra == all columns in df) and delim
        if isinstance(op, TransformOperator):
            fin_tar_cols, fin_extra_cols = op.out_columns(target_cols, extra_cols, self.delim)
            tup_rep = target_cols, extra_cols, op, parent, fin_tar_cols, fin_extra_cols
        # for stat ops, which do not change data in columns
        if not tup_rep:
            tup_rep = target_cols, extra_cols, op, parent, target_cols, extra_cols
        self.columns_ctx["full"][op._id] = tup_rep
        return tup_rep

    def _get_op_count(self, op_id):
        return sum(1 for op in self.ops_in if op_id in op)

    def _create_phases(self):
        # create new ordering based on placement and full_dict keys list
        ordered_ops = self._find_order()
        phases = []
        excess = ordered_ops
        while excess:
            phase, excess = self._create_phase(excess)
            phases.append(phase)
        return phases

    def _find_order(self):
        ops_ordered = []
        ops_origin = list(self.columns_ctx["full"].keys()).copy()
        ops_not_added = ops_origin[1:]
        for op_focus in ops_origin:
            ops_added, ops_not_added = self._find_order_single(op_focus, ops_not_added)
            ops_ordered.append(ops_added)
        ops_ordered.append(ops_not_added)
        res_list = []
        for ops_set in ops_ordered:
            res_list = res_list + ops_set
        return res_list

    def _find_order_single(self, op_focus, op_ids):
        op_ordered, not_added, parents_ref = [], [], []
        for op_id in op_ids:
            place = self.placement[op_id]
            # k is op._id, v is all finds... need second to last
            if len(place) > 1:
                op_after, idx = place[-2]
            else:
                op_after, idx = place[-1]
            op_task = self.columns_ctx["full"][op_id]
            target_cols, extra_cols, op, parent, fin_tar_cols, fin_extra_cols = op_task
            if parent:
                parents_ref.append(parent)
            if op_after in op_focus and op_id not in parents_ref and op_after not in parents_ref:
                # op has no requirements move to front
                op_ordered.append(op_id)
            else:
                not_added.append(op_id)
        return op_ordered, not_added

    def _create_phase(self, op_ordered):
        # given the correctly ordered op_task list (full_dict),
        # decide index splits for individual phases
        parents_ref = []
        for idx, op_id in enumerate(op_ordered):
            target_cols, extra_cols, op, parent, fin_tar_cols, fin_extra_cols = self.columns_ctx[
                "full"
            ][op_id]
            if op._id in parents_ref:
                return op_ordered[:idx], op_ordered[idx:]
            if parent:
                parents_ref.append(parent)

        return op_ordered, []

    def _get_target_cols(self, operators):
        # all operators in a list are chained therefore based on parent in list
        if type(operators) is list:
            target_cols = operators[0].get_default_in()
        else:
            target_cols = operators.get_default_in()
        return target_cols

    def _config_add_ops(self, operators, phase):
        """
        This function serves to translate the operator list api into backend
        ready dependency dictionary.

        Parameters
        ----------
        operators: list
            list of operators or single operator to be added into the
            preprocessing phase
        phase:
            identifier for feature engineering FE or preprocessing PP
        """
        target_cols = self._get_target_cols(operators)
        if not target_cols or (
            target_cols in self.columns_ctx and not self.columns_ctx[target_cols]["base"]
        ):
            warnings.warn(f"Did not add operators: {operators}, target columns is empty.")
            return
        if phase in self.config and target_cols in self.config[phase]:
            for op in operators:
                self._check_op_count(op)
            self.config[phase][target_cols].append(operators)
            return

        warnings.warn(f"No main key {phase} or sub key {target_cols} found in config")

    def op_default_check(self, operators, default_in):
        if not type(operators) is list:
            operators = [operators]
        for op in operators:
            if op.default_in != default_in and op.default_in != "all":
                warnings.warn(
                    f"{op._id} was not added. This op is not designed for use"
                    f" with {default_in} columns"
                )
                operators.remove(op)
        return operators

    def add_feature(self, operators):
        """
        Adds feature engineering operator(s), while mapping
        to the correct columns given operator dependencies.

        Parameters
        -----------
        operators : object
            list of operators or single operator to be
            added into the feature engineering phase
        """
        if not isinstance(operators, list):
            operators = [operators]
        self._config_add_ops(operators, "FE")

    def add_cat_feature(self, operators):
        """
        Adds categorical feature engineering operator(s), while mapping
        to the correct columns given operator dependencies.

        Parameters
        -----------
        operators : object
            list of categorical operators or single operator to be
            added into the feature engineering phase
        """

        operators = self.op_default_check(operators, "categorical")
        if operators:
            self.add_feature(operators)

    def add_cont_feature(self, operators):

        """
        Adds continuous feature engineering operator(s)
        to the workflow.

        Parameters
        -----------
        operators : object
            continuous objects such as FillMissing, Clip and LogOp
        """

        operators = self.op_default_check(operators, "continuous")
        if operators:
            self.add_feature(operators)

    def add_cat_preprocess(self, operators):

        """
        Adds categorical pre-processing operator(s)
        to the workflow.

        Parameters
        -----------
        operators : object
            categorical objects such as Categorify
        """

        operators = self.op_default_check(operators, "categorical")
        if operators:
            self.add_preprocess(operators)

    def add_cont_preprocess(self, operators):

        """
        Adds continuous pre-processing operator(s)
        to the workflow.

        Parameters
        -----------
        operators : object
            continuous objects such as Normalize
        """

        operators = self.op_default_check(operators, "continuous")
        if operators:
            self.add_preprocess(operators)

    def add_preprocess(self, operators):

        """
        Adds preprocessing operator(s), while mapping
        to the correct columns given operator dependencies.

        Parameters
        -----------
        operators : object
            list of operators or single operator, Op/s to be
            added into the preprocessing phase
        """
        # must add last operator from FE for get_default_in
        target_cols = self._get_target_cols(operators)
        if self.config["FE"][target_cols]:
            op_to_add = self.config["FE"][target_cols][-1]
        else:
            op_to_add = []
        if type(op_to_add) is list and op_to_add:
            op_to_add = op_to_add[-1]
        if op_to_add:
            op_to_add = [op_to_add]
        if type(operators) is list:
            op_to_add = op_to_add + operators
        else:
            op_to_add.append(operators)
        self._config_add_ops(op_to_add, "PP")

    def finalize(self):
        """
        When using operator list api, this allows the user to declare they
        have finished adding all operators and are ready to start processing
        data.
        """
        self.load_config(self.config)

    def load_config(self, config, pro=False):
        """
        This function extracts all the operators from the given phases and produces a
        set of phases with necessary operators to complete configured pipeline.

        Parameters
        ----------
        config : dict
            this object contains the phases and user specified operators
        pro: bool
            signals if config should be parsed via dependency dictionary or
            operator list api
        """
        # separate FE and PP
        if not pro:
            config = self._compile_dict_from_list(config)
        task_sets = {}
        master_task_list = []
        for task_set in config.keys():
            task_sets[task_set] = self._build_tasks(config[task_set], task_set, master_task_list)
            master_task_list = master_task_list + task_sets[task_set]

        self._register_ops(master_task_list.copy())
        phases = self._create_phases()
        self.phases = self.translate(master_task_list, phases)
        self._create_final_col_refs(task_sets)

    def translate(self, mtl, phases):
        real_phases = []
        for phase in phases:
            real_phase = []
            for op_id in phase:
                for op_task in mtl:
                    op = op_task[0]
                    if op._id == op_id:
                        real_phase.append(op_task)
                        break
            real_phases.append(real_phase)
        return real_phases

    def _compile_dict_from_list(self, config):
        """
        This function retrieves all the operators from the different keys in
        the config object.

        Parameters
        -----------
        config : dict
            this dictionary has phases(key) and the corresponding list of operators for
            each phase.
        """
        ret = {}
        for phase, task_list in config.items():
            ret[phase] = {}
            for k, v in task_list.items():
                tasks = []
                for obj in v:
                    if not isinstance(obj, collections.abc.Sequence):
                        obj = [obj]
                    for idx, op in enumerate(obj):
                        tasks.append((op, [obj[idx - 1]._id] if idx > 0 else []))

                ret[phase][k] = tasks
        return ret

    def _create_final_col_refs(self, task_sets):
        """
        This function creates a reference of all the operators whose produced
        columns will be available in the final set of columns. First step in
        creating the final columns list.
        """

        if "final" in self.columns_ctx.keys():
            return
        final = {}
        # all preprocessing tasks have a parent operator, it could be None
        # task (operator, main_columns_class, col_sub_key,  required_operators)
        for task in task_sets["PP"]:
            # an operator cannot exist twice
            if not task[1] in final.keys():
                final[task[1]] = []
            # detect incorrect dependency loop
            for x in final[task[1]]:
                if x in task[2]:
                    final[task[1]].remove(x)
            # stats dont create columns so id would not be in columns ctx
            if not task[0].__class__.__base__ == StatOperator:
                final[task[1]].append(task[0]._id)
        # add labels too specific because not specifically required in init
        final["label"] = []
        for col_ctx in self.columns_ctx["label"].values():
            if not final["label"]:
                final["label"] = ["base"]
            else:
                final["label"] = final["label"] + col_ctx
        # if no operators run in preprocessing we grab base columns
        if "continuous" not in final:
            # set base columns
            final["continuous"] = ["base"]
        if "categorical" not in final:
            final["categorical"] = ["base"]
        if "all" not in final:
            final["all"] = ["base"]
        self.columns_ctx["final"] = {}
        self.columns_ctx["final"]["ctx"] = final

    def create_final_cols(self):
        """
        This function creates an entry in the columns context dictionary,
        not the references to the operators. In this method we detail all
        operator references with actual column names, and create a list.
        The entry represents the final columns that should be in finalized
        dataframe.
        """
        # still adding double need to stop that
        final_ctx = {}
        for key, ctx_list in self.columns_ctx["final"]["ctx"].items():
            to_add = None
            for ctx in ctx_list:
                if ctx not in self.columns_ctx[key].keys():
                    ctx = "base"
                to_add = (
                    self.columns_ctx[key][ctx]
                    if not to_add
                    else to_add + self.columns_ctx[key][ctx]
                )
            if key not in final_ctx.keys():
                final_ctx[key] = to_add
            else:
                final_ctx[key] = final_ctx[key] + to_add
        self.columns_ctx["final"]["cols"] = final_ctx

    def get_final_cols_names(self, col_type):
        """
        Returns all the column names after preprocessing and feature
        engineering.

        Parameters
        -----------
        col_type : str
        """
        col_names = []
        for c_names in self.columns_ctx[col_type].values():
            for name in c_names:
                if name not in col_names:
                    col_names.append(name)
        return col_names

    def _build_tasks(self, task_dict: dict, task_set, master_task_list):
        """
        task_dict: the task dictionary retrieved from the config
        Based on input config information
        """
        # task format = (operator, main_columns_class, col_sub_key,  required_operators, parent)
        dep_tasks = []
        for cols, task_list in task_dict.items():
            for target_op, dep_grp in task_list:
                if isinstance(target_op, DFOperator):
                    # check that the required stat is grabbed
                    # for all necessary parents
                    for opo in target_op.req_stats:
                        self._check_op_count(opo)
                        self.ops_in.append(opo._id)
                        dep_grp = dep_grp if dep_grp else ["base"]
                        dep_tasks.append((opo, cols, dep_grp, [], target_op))
                # after req stats handle target_op
                self.ops_in.append(target_op._id)
                dep_grp = dep_grp if dep_grp else ["base"]
                req_ops = [] if not hasattr(target_op, "req_stats") else target_op.req_stats
                dep_tasks.append((target_op, cols, dep_grp, req_ops, []))
        return dep_tasks

    def _run_trans_ops_for_phase(self, gdf, tasks):
        for task in tasks:
            op, cols_grp, target_cols, _, _ = task
            if isinstance(op, DFOperator):
                gdf = op.apply_op(gdf, self.columns_ctx, cols_grp, target_cols, self.stats)
            elif isinstance(op, TransformOperator):
                gdf = op.apply_op(gdf, self.columns_ctx, cols_grp, target_cols=target_cols)
        return gdf

    def apply_ops(
        self, gdf, start_phase=None, end_phase=None, writer=None, output_path=None, dtypes=None
    ):
        """
        gdf: cudf dataframe
        Controls the application of registered preprocessing phase op
        tasks, can only be used after apply has been performed
        """
        # put phases that you want to run represented in a slice
        # dont run stat_ops in apply
        # run the PP ops
        start = start_phase if start_phase else 0
        end = end_phase if end_phase else len(self.phases)
        for phase_index in range(start, end):
            start = time.time()
            gdf = self._run_trans_ops_for_phase(gdf, self.phases[phase_index])
            self.timings["preproc_apply"] += time.time() - start
            if phase_index == len(self.phases) - 1 and writer and output_path:

                if writer.need_cal_col_names:
                    cat_names = self.get_final_cols_names("categorical")
                    cont_names = self.get_final_cols_names("continuous")
                    label_names = self.get_final_cols_names("label")
                    writer.set_col_names(labels=label_names, cats=cat_names, conts=cont_names)
                    writer.need_cal_col_names = False
                start_write = time.time()
                # Special dtype conversion
                gdf = _set_dtypes(gdf, dtypes)
                writer.add_data(gdf)
                self.timings["write_df"] += time.time() - start_write
        return gdf

    def _update_statistics(self, stat_op):
        stats = [stat for stat in stat_op.registered_stats() if stat in self.stats.keys()]
        if not stats:
            # add if doesnt exist
            self.stats.update(stat_op.stats_collected())
        else:
            # if it does exist, append to it
            for key, val in stat_op.stats_collected():
                self.stats[key].update(val)

    def save_stats(self, path):
        main_obj = {}
        stats_drop = {}
        for name, stat in self.stats.items():
            if name not in stats_drop.keys():
                stats_drop[name] = stat
        main_obj["stats"] = stats_drop
        main_obj["columns_ctx"] = {}
        for key in self.columns_ctx.keys():
            if "full" != key:
                main_obj["columns_ctx"][key] = self.columns_ctx[key]
        self.columns_ctx
        with open(path, "w") as outfile:
            yaml.safe_dump(main_obj, outfile, default_flow_style=False)

    def load_stats(self, path):
        def _set_stats(self, stats_dict):
            for key, stat in stats_dict.items():
                self.stats[key] = stat

        with open(path, "r") as infile:
            main_obj = yaml.safe_load(infile)
            _set_stats(self, main_obj["stats"])
            self.columns_ctx = main_obj["columns_ctx"]

    def clear_stats(self):
        self.stats = {}


def get_new_config():
    """
    boiler config object, to be filled in with targeted operator tasks
    """
    config = {}
    config["FE"] = {}
    config["FE"]["all"] = []
    config["FE"]["continuous"] = []
    config["FE"]["categorical"] = []
    config["PP"] = {}
    config["PP"]["all"] = []
    config["PP"]["continuous"] = []
    config["PP"]["categorical"] = []
    return config


class Workflow(BaseWorkflow):
    """
    Dask-based NVTabular Workflow Class
    """

    def __init__(self, client=None, **kwargs):
        super().__init__(**kwargs)
        self.ddf = None
        self.client = client
        self._shuffle_parts = False
        self._base_phase = 0

    def set_ddf(self, ddf, shuffle=None):
        if isinstance(ddf, (dask_cudf.DataFrame, Dataset)):
            self.ddf = ddf
            if shuffle is not None:
                self._shuffle_parts = shuffle
        else:
            raise TypeError("ddf type not supported.")

    def get_ddf(self):
        if self.ddf is None:
            raise ValueError("No dask_cudf frame available.")
        elif isinstance(self.ddf, Dataset):
            # Right now we can't distinguish between input columns and generated columns
            # in the dataset, we don't limit the columm set right now in the to_ddf call
            # (https://github.com/NVIDIA/NVTabular/issues/409 )
            return self.ddf.to_ddf(shuffle=self._shuffle_parts)
        return self.ddf

    @staticmethod
    def _aggregated_op(gdf, ops):
        for op in ops:
            columns_ctx, cols_grp, target_cols, logic, stats_context = op
            gdf = logic(gdf, columns_ctx, cols_grp, target_cols, stats_context)
        return gdf

    def _aggregated_dask_transform(self, ddf, transforms):
        # Assuming order of transforms corresponds to dependency ordering
        meta = ddf._meta
        for transform in transforms:
            columns_ctx, cols_grp, target_cols, logic, stats_context = transform
            meta = logic(meta, columns_ctx, cols_grp, target_cols, stats_context)
        return ddf.map_partitions(self.__class__._aggregated_op, transforms, meta=meta)

    def exec_phase(self, phase_index, record_stats=True, update_ddf=True):
        """
        Gather necessary column statistics in single pass.
        Execute statistics for one phase only (given by phase index),
        but (laxily) perform all transforms for current and previous phases.
        """
        transforms = []

        # Need to perform all transforms up to, and including,
        # the current phase (not only the current phase).  We do this
        # so that we can avoid persisitng intermediate transforms
        # needed for statistics.
        phases = range(self._base_phase, phase_index + 1)
        for ind in phases:
            for task in self.phases[ind]:
                op, cols_grp, target_cols, _, _ = task
                if isinstance(op, TransformOperator):
                    stats_context = self.stats if isinstance(op, DFOperator) else None
                    logic = op.apply_op
                    transforms.append(
                        (self.columns_ctx, cols_grp, target_cols, logic, stats_context)
                    )
                elif not isinstance(op, StatOperator):
                    raise ValueError("Unknown Operator Type")

        # Perform transforms as single dask task (per ddf partition)
        _ddf = self.get_ddf()
        if transforms:
            _ddf = self._aggregated_dask_transform(_ddf, transforms)

        stats = []
        if record_stats:
            for task in self.phases[phase_index]:
                op, cols_grp, target_cols, _, _ = task
                if isinstance(op, StatOperator):
                    stats.append((op.stat_logic(_ddf, self.columns_ctx, cols_grp, target_cols), op))
                    # TODO: Don't want to update the internal ddf here if we can
                    # avoid it.  It may be better to just add the new column?
                    if op._ddf_out is not None:
                        self.set_ddf(op._ddf_out)
                        # We are updating the internal `ddf`, so we shouldn't
                        # redo transforms up to this phase in later phases.
                        self._base_phase = phase_index

        # Compute statistics if necessary
        if stats:
            if self.client:
                for r in self.client.compute(stats):
                    computed_stats, op = r.result()
                    op.finalize(computed_stats)
                    self._update_statistics(op)
                    op.clear()
            else:
                for r in dask.compute(stats, scheduler="synchronous")[0]:
                    computed_stats, op = r
                    op.finalize(computed_stats)
                    self._update_statistics(op)
                    op.clear()
            del stats

        # Update interal ddf.
        # Cancel futures and delete _ddf if allowed.
        if transforms and update_ddf:
            self.set_ddf(_ddf)
        else:
            if self.client:
                self.client.cancel(_ddf)
            del _ddf

    def apply(
        self,
        dataset,
        apply_offline=True,
        record_stats=True,
        shuffle=None,
        output_path="./ds_export",
        output_format="parquet",
        out_files_per_proc=None,
        num_io_threads=0,
        dtypes=None,
    ):
        """
        Runs all the preprocessing and feature engineering operators.
        Also, shuffles the data if a `shuffle` option is specified.

        Parameters
        -----------
        dataset : object
        apply_offline : boolean
            Runs operators in offline mode or not
        record_stats : boolean
            Record the stats in file or not. Only available
            for apply_offline=True
        shuffle : nvt.io.Shuffle enum
            How to shuffle the output dataset. Shuffling is only
            performed if the data is written to disk. For all options,
            other than `None` (which means no shuffling), the partitions
            of the underlying dataset/ddf will be randomly ordered. If
            `PER_PARTITION` is specified, each worker/process will also
            shuffle the rows within each partition before splitting and
            appending the data to a number (`out_files_per_proc`) of output
            files. Output files are distinctly mapped to each worker process.
            If `PER_WORKER` is specified, each worker will follow the same
            procedure as `PER_PARTITION`, but will re-shuffle each file after
            all data is persisted.  This results in a full shuffle of the
            data processed by each worker.  To improve performace, this option
            currently uses host-memory `BytesIO` objects for the intermediate
            persist stage. The `FULL` option is not yet implemented.
        output_path : string
            Path to write processed/shuffled output data
        output_format : {"parquet", "hugectr", None}
            Output format to write processed/shuffled data. If None,
            no output dataset will be written (and shuffling skipped).
        out_files_per_proc : integer
            Number of files to create (per process) after
            shuffling the data
        num_io_threads : integer
            Number of IO threads to use for writing the output dataset.
            For `0` (default), no dedicated IO threads will be used.
        dtypes : dict
            Dictionary containing desired datatypes for output columns.
            Keys are column names, values are datatypes.
        """

        # Check shuffle argument
        shuffle = _check_shuffle_arg(shuffle)

        # If no tasks have been loaded then we need to load internal config
        if not self.phases:
            self.finalize()

        # Gather statstics (if apply_offline), and/or transform
        # and write out processed data
        if apply_offline:
            self.build_and_process_graph(
                dataset,
                output_path=output_path,
                record_stats=record_stats,
                shuffle=shuffle,
                output_format=output_format,
                out_files_per_proc=out_files_per_proc,
                num_io_threads=num_io_threads,
                dtypes=dtypes,
            )
        else:
            self.iterate_online(
                dataset,
                output_path=output_path,
                shuffle=shuffle,
                output_format=output_format,
                out_files_per_proc=out_files_per_proc,
                num_io_threads=num_io_threads,
                dtypes=dtypes,
            )

    def iterate_online(
        self,
        dataset,
        end_phase=None,
        output_path=None,
        shuffle=None,
        output_format=None,
        out_files_per_proc=None,
        apply_ops=True,
        num_io_threads=0,
        dtypes=None,
    ):
        """Iterate through dataset and (optionally) apply/shuffle/write."""
        # Check shuffle argument
        shuffle = _check_shuffle_arg(shuffle)

        # Check if we have a (supported) writer
        output_path = output_path or "./"
        output_path = str(output_path)
        writer = writer_factory(
            output_format,
            output_path,
            out_files_per_proc,
            shuffle,
            bytes_io=(shuffle == Shuffle.PER_WORKER),
            num_threads=num_io_threads,
        )

        # Iterate through dataset, apply ops, and write out processed data
        if apply_ops:
            columns = self.columns_ctx["all"]["base"]
            for gdf in dataset.to_iter(shuffle=(shuffle is not None), columns=columns):
                self.apply_ops(gdf, output_path=output_path, writer=writer, dtypes=dtypes)

        # Close writer and write general/specialized metadata
        if writer:
            general_md, special_md = writer.close()

            # Note that we "could" have the special and general metadata
            # written during `writer.close()` (just above) for the single-GPU case.
            # Instead, the metadata logic is separated from the `Writer` object to
            # simplify multi-GPU integration. When using Dask, we cannot assume
            # that the "shared" metadata files can/will be written by the same
            # process that writes the data.
            writer.write_special_metadata(special_md, writer.fs, output_path)
            writer.write_general_metadata(general_md, writer.fs, output_path)

    def update_stats(self, dataset, end_phase=None):
        """Collect statistics only."""
        self.build_and_process_graph(dataset, end_phase=end_phase, record_stats=True)

    def build_and_process_graph(
        self,
        dataset,
        end_phase=None,
        output_path=None,
        record_stats=True,
        shuffle=None,
        output_format=None,
        out_files_per_proc=None,
        apply_ops=True,
        num_io_threads=0,
        dtypes=None,
    ):
        """Build Dask-task graph for workflow.

        Full graph is only executed if `output_format` is specified.
        """
        # Check shuffle argument
        shuffle = _check_shuffle_arg(shuffle)

        end = end_phase if end_phase else len(self.phases)

        if output_format not in ("parquet", "hugectr", None):
            raise ValueError(f"Output format {output_format} not yet supported with Dask.")

        # Clear worker caches to be "safe"
        if self.client:
            self.client.run(clean_worker_cache)
        else:
            clean_worker_cache()

        self.set_ddf(dataset, shuffle=(shuffle is not None))
        if apply_ops:
            self._base_phase = 0  # Set _base_phase
            for idx, _ in enumerate(self.phases[:end]):
                self.exec_phase(idx, record_stats=record_stats, update_ddf=(idx == (end - 1)))
            self._base_phase = 0  # Re-Set _base_phase

        if dtypes:
            ddf = self.get_ddf()
            _meta = _set_dtypes(ddf._meta, dtypes)
            self.set_ddf(ddf.map_partitions(_set_dtypes, dtypes, meta=_meta))

        if output_format:
            output_path = output_path or "./"
            output_path = str(output_path)
            self.ddf_to_dataset(
                output_path,
                output_format=output_format,
                shuffle=shuffle,
                out_files_per_proc=out_files_per_proc,
                num_threads=num_io_threads,
            )

    def write_to_dataset(
        self,
        path,
        dataset,
        apply_ops=False,
        out_files_per_proc=None,
        shuffle=None,
        output_format="parquet",
        iterate=False,
        nfiles=None,
        num_io_threads=0,
        dtypes=None,
    ):
        """Write data to shuffled parquet dataset.

        Assumes statistics are already gathered.
        """
        # Check shuffle argument
        shuffle = _check_shuffle_arg(shuffle)

        if nfiles:
            warnings.warn("nfiles is deprecated. Use out_files_per_proc")
            if out_files_per_proc is None:
                out_files_per_proc = nfiles
        out_files_per_proc = out_files_per_proc or 1

        path = str(path)
        if iterate:
            self.iterate_online(
                dataset,
                output_path=path,
                shuffle=shuffle,
                output_format=output_format,
                out_files_per_proc=out_files_per_proc,
                apply_ops=apply_ops,
                num_io_threads=num_io_threads,
                dtypes=dtypes,
            )
        else:
            self.build_and_process_graph(
                dataset,
                output_path=path,
                record_stats=False,
                shuffle=shuffle,
                output_format=output_format,
                out_files_per_proc=out_files_per_proc,
                apply_ops=apply_ops,
                num_io_threads=num_io_threads,
                dtypes=dtypes,
            )

    def ddf_to_dataset(
        self,
        output_path,
        shuffle=None,
        out_files_per_proc=None,
        output_format="parquet",
        num_threads=0,
    ):
        """Dask-based dataset output.

        Currently supports parquet only.
        """
        if output_format not in ("parquet", "hugectr"):
            raise ValueError("Only parquet/hugectr output supported with Dask.")
        ddf = self.get_ddf()
        fs = get_fs_token_paths(output_path)[0]
        fs.mkdirs(output_path, exist_ok=True)
        if shuffle or out_files_per_proc:

            cat_names = self.get_final_cols_names("categorical")
            cont_names = self.get_final_cols_names("continuous")
            label_names = self.get_final_cols_names("label")

            # Output dask_cudf DataFrame to dataset
            _ddf_to_dataset(
                ddf,
                fs,
                output_path,
                shuffle,
                out_files_per_proc,
                cat_names,
                cont_names,
                label_names,
                output_format,
                self.client,
                num_threads,
            )
            return

        # Default (shuffle=None and out_files_per_proc=None)
        # Just use `dask_cudf.to_parquet`
        fut = ddf.to_parquet(output_path, compression=None, write_index=False, compute=False)
        if self.client is None:
            fut.compute(scheduler="synchronous")
        else:
            fut.compute()
