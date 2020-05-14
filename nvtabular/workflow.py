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
import logging
import os
import time
import warnings

import cudf
import yaml
from cudf._lib.nvtx import annotate

from nvtabular.ds_writer import DatasetWriter
from nvtabular.encoder import DLLabelEncoder
from nvtabular.io import HugeCTR, Shuffler
from nvtabular.ops import DFOperator, Export, StatOperator, TransformOperator, all_ops

try:
    import cupy as cp
except ImportError:
    import numpy as cp


LOG = logging.getLogger("nvtabular")


class Workflow:

    """
    Workflow class organizes and runs all the feature engineering
    and pre-processing operators for your workflow.

    Parameters
    -----------
    cat_names : list of str
        Names of the categorical columns.
    cont_names : list of str
        Names of the continous columns.
    label_name : list of str
        Names of the label column.
    feat_ops : list of feature engineering operator objects
        Feature engineering ops objects GroupBy.
        New feature engineering operators can be added
        later using the other functions
    stat_ops : list of stats operator objects
        Feature engineering ops objects such as
        Categorify and Normalize. New feature engineering
        operators can be added later using the other functions.
    df_ops : list of data frame operator objects
    to_cpu : bool, default True
    config : bool
    export : bool, default False
    export_path : str, default "./ds_export"
    """

    def __init__(
        self,
        cat_names=None,
        cont_names=None,
        label_name=None,
        feat_ops=None,
        stat_ops=None,
        df_ops=None,
        to_cpu=True,
        config=None,
        export=False,
        export_path="./ds_export",
    ):
        self.reg_funcs = {
            StatOperator: self.reg_stat_ops,
            TransformOperator: self.reg_feat_ops,
            DFOperator: self.reg_df_ops,
        }
        self.master_task_list = []
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
        self.feat_ops = {}
        self.stat_ops = {}
        self.df_ops = {}
        self.stats = {}
        self.task_sets = {}
        self.ds_exports = export_path
        self.to_cpu = to_cpu
        self.export = export
        self.ops_args = {}
        self.current_file_num = 0
        self.timings = {
            "shuffle_df": 0.0,
            "shuffle_fin": 0.0,
            "preproc_apply": 0.0,
            "preproc_reapply": 0.0,
        }
        if config:
            self.load_config(config)
        else:
            # create blank config and for later fill in
            self.config = get_new_list_config()

        self.clear_stats()

    def get_tar_cols(self, operators):
        # all operators in a list are chained therefore based on parent in list
        if type(operators) is list:
            target_cols = operators[0].get_default_in()
        else:
            target_cols = operators.get_default_in()
        return target_cols

    def config_add_ops(self, operators, phase):
        """
        This function serves to translate the operator list api into backend
        ready dependency dictionary.

        Parameters
        ----------
        operators: list
            list of operators or single operator, Op/s to be added into the
            preprocessing phase
        phase:
            identifier for feature engineering FE or preprocessing PP
        """
        target_cols = self.get_tar_cols(operators)
        if phase in self.config and target_cols in self.config[phase]:
            # append operator as single ent1ry or as a list
            # maybe should be list always to make downstream easier
            self.config[phase][target_cols].append(operators)
            return

        warnings.warn(f"No main key {phase} or sub key {target_cols} found in config")

    def op_default_check(self, operators, default_in):
        if not type(operators) is list:
            operators = [operators]
        for op in operators:
            if op.default_in not in default_in:
                warnings.warn(
                    f"{op._id} was not added. This op is not designed for use"
                    f" with {default_in} columns"
                )
                operators.remove(op)

    def add_feature(self, operators):
        """
        Adds feature engineering operator(s), while mapping
        to the correct columns given operator dependencies.

        Parameters
        -----------
        operators : object
            list of operators or single operator, Op/s to be
            added into the feature engineering phase
        """

        self.config_add_ops(operators, "FE")

    def add_cat_feature(self, operators):
        """
        Adds categorical feature engineering operator(s), while mapping
        to the correct columns given operator dependencies.

        Parameters
        -----------
        operators : object
            list of categorical operators or single operator, Op/s to be
            added into the feature engineering phase
        """

        self.op_default_check(operators, "categorical")
        if operators:
            self.add_feature(operators)

    def add_cont_feature(self, operators):

        """
        Adds continuous feature engineering operator(s)
        to the workflow.

        Parameters
        -----------
        operators : object
            continuous objects such as ZeroFill and LogOp
        """

        self.op_default_check(operators, "continuous")
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

        self.op_default_check(operators, "categorical")
        if operators:
            self.add_preprocess(operators)

    def add_cont_preprocess(self, operators):

        """
        Adds continuous pre-processing operator(s)
        to the workflow.

        Parameters
        -----------
        operators : object
            categorical objects such as Normalize
        """

        self.op_default_check(operators, "continuous")
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
        target_cols = self.get_tar_cols(operators)
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
        self.config_add_ops(op_to_add, "PP")

    def reg_all_ops(self, task_list):
        for tup in task_list:
            self.reg_funcs[tup[0].__class__.__base__]([tup[0]])

    def finalize(self):
        """
        When using operator list api, this allows the user to declare they
        have finished adding all operators and are ready to start processing
        data.
        """
        self.load_config(self.config)

    def reg_feat_ops(self, feat_ops):
        """
        Register Feature engineering operators
        """
        for feat_op in feat_ops:
            self.feat_ops[feat_op._id] = feat_op

    def reg_df_ops(self, df_ops):
        """
        Register preprocessing operators
        """
        for df_op in df_ops:
            dfop_id, dfop_rs = df_op._id, df_op.req_stats
            self.reg_stat_ops(dfop_rs)
            self.df_ops[dfop_id] = df_op

    def reg_stat_ops(self, stat_ops):
        """
        Register statistical operators
        """
        for stat_op in stat_ops:
            # pull stats, ensure no duplicates
            for stat in stat_op.registered_stats():
                if stat not in self.stats:
                    self.stats[stat] = {}
            # add actual statistic operator, after all stats added
            self.stat_ops[stat_op._id] = stat_op

    def write_to_dataset(self, path, itr, apply_ops=False, nfiles=1, shuffle=True, **kwargs):
        """ Write data to shuffled parquet dataset.
        """
        writer = DatasetWriter(path, nfiles=nfiles)

        for gdf in itr:
            if apply_ops:
                gdf = self.apply_ops(gdf)
            writer.write(gdf, shuffle=shuffle)
        writer.write_metadata()
        return

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
            config = self.compile_dict_from_list(config)
        self.task_sets = {}
        for task_set in config.keys():
            self.task_sets[task_set] = self.build_tasks(config[task_set], task_set)
            self.master_task_list = self.master_task_list + self.task_sets[task_set]

        self.reg_all_ops(self.master_task_list)
        baseline, leftovers = self.sort_task_types(self.master_task_list)
        self.phases.append(baseline)
        self.phase_creator(leftovers)
        # check if export wanted
        if self.export:
            self.phases_export()
        self.create_final_col_refs()

    def phase_creator(self, task_list):
        """
        task_list: list, phase specific list of operators and dependencies
        ---
        This function splits the operators in the task list and adds in any
        dependent operators i.e. statistical operators required by selected
        operators.
        """
        for task in task_list:
            added = False

            cols_needed = task[2].copy()
            if "base" in cols_needed:
                cols_needed.remove("base")
            for idx, phase in enumerate(self.phases):
                if added:
                    break
                for p_task in phase:
                    if not cols_needed:
                        break
                    if p_task[0]._id in cols_needed:
                        cols_needed.remove(p_task[0]._id)
                if not cols_needed and self.find_parents(task[3], idx):
                    added = True
                    phase.append(task)

            if not added:
                self.phases.append([task])

    def phases_export(self):
        """
        Export each phase from the dependency dictionary, that creates transformations.
        """
        for idx, phase in enumerate(self.phases[:-1]):
            trans_op = False
            # only export if the phase has a transform operator on the dataset
            # otherwise all stats will be saved in the tabular object
            for task in phase:
                if isinstance(task[0], TransformOperator):
                    trans_op = True
                    break
            if trans_op:
                tar_path = os.path.join(self.ds_exports, str(idx))
                phase.append([Export(path=f"{tar_path}"), None, [], []])

    def find_parents(self, ops_list, phase_idx):
        """
        Attempt to find all ops in ops_list within subrange of phases
        """
        ops_copy = ops_list.copy()
        for op in ops_list:
            for phase in self.phases[:phase_idx]:
                if not ops_copy:
                    break
                for task in phase:
                    if not ops_copy:
                        break
                    if op._id in task[0]._id:
                        ops_copy.remove(op)
        if not ops_copy:
            return True

    def sort_task_types(self, master_list):
        """
        This function helps ordering and breaking up the master list of operators into the
        correct phases.

        Parameters
        -----------
        master_list : list
            a complete list of all necessary operators to complete specified pipeline
        """
        nodeps = []
        for tup in master_list:
            if "base" in tup[2]:
                # base feature with no dependencies
                if not tup[3]:
                    master_list.remove(tup)
                    nodeps.append(tup)
        return nodeps, master_list

    def compile_dict_from_list(self, task_list_dict):
        """
        This function retrieves all the operators from the different keys in
        the task_list_dict object.

        Parameters
        -----------
        task_list_dict : dict
            this dictionary has phases(key) and the corresponding list of operators for
            each phase.
        """
        tk_d = {}
        phases = 0
        for phase, task_list in task_list_dict.items():
            tk_d[phase] = {}
            for k, v in task_list.items():
                tk_d[phase][k] = self.extract_tasks_dict(v)
            # increment at end for next if exists
            phases = phases + 1
        return tk_d

    def extract_tasks_dict(self, task_list):
        """
        The function serves as a shim that can turn lists of operators
        into the dictionary dependency format required for processing.
        """
        # contains a list of lists [[fillmissing, Logop]], Normalize, Categorify]
        task_dicts = []
        for obj in task_list:
            if isinstance(obj, list):

                for idx, op in enumerate(obj):
                    # kwargs for mapping during load later
                    self.ops_args[op._id] = op.export_op()[op._id]
                    if idx > 0:
                        to_add = {op._id: [[obj[idx - 1]._id]]}
                    else:
                        to_add = {op._id: [[]]}
                    task_dicts.append(to_add)
            else:
                self.ops_args[obj._id] = obj.export_op()[obj._id]
                to_add = {obj._id: [[]]}
                task_dicts.append(to_add)
        return task_dicts

    def create_final_col_refs(self):
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
        for task in self.task_sets["PP"]:
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
        for p_set, col_ctx in self.columns_ctx["label"].items():
            if not final["label"]:
                final["label"] = col_ctx
            else:
                final["label"] = final["label"] + col_ctx
        # if no operators run in preprocessing we grab base columns
        if "continuous" not in final:
            # set base columns
            final["continuous"] = self.columns_ctx["continuous"]["base"]
        if "categorical" not in final:
            final["categorical"] = self.columns_ctx["categorical"]["base"]
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

    def build_tasks(self, task_dict: dict, task_set):
        """
        task_dict: the task dictionary retrieved from the config
        Based on input config information
        """
        # task format = (operator, main_columns_class, col_sub_key,  required_operators)
        dep_tasks = []
        for cols, task_list in task_dict.items():
            for task in task_list:
                for op_id, dep_set in task.items():
                    # get op from op_id
                    # operators need to be instantiated with state information
                    target_op = all_ops[op_id](**self.ops_args[op_id])
                    if dep_set:
                        for dep_grp in dep_set:
                            # handle required stats of target op on
                            # all the dependent columns
                            for dep in dep_grp:
                                if task_set in "PP" and not self.op_preprocess(dep):
                                    dep_grp.remove(dep)
                            if hasattr(target_op, "req_stats"):
                                # check that the required stat is grabbed
                                # for all necessary parents
                                for opo in target_op.req_stats:
                                    # only add if it doesnt already exist=
                                    if not self.is_repeat_op(opo, cols):
                                        dep_grp = dep_grp if dep_grp else ["base"]
                                        dep_tasks.append((opo, cols, dep_grp, []))
                            # after req stats handle target_op
                            dep_grp = dep_grp if dep_grp else ["base"]
                            parents = (
                                [] if not hasattr(target_op, "req_stats") else target_op.req_stats
                            )
                            if not self.is_repeat_op(target_op, cols):
                                dep_tasks.append((target_op, cols, dep_grp, parents))
        return dep_tasks

    def op_preprocess(self, target_op_id):
        # find operator given id
        target_op = self.find_op(target_op_id)
        # check if operator has preprocessing
        # if preprocessing, break
        if hasattr(target_op, "preprocessing"):
            return target_op.preprocessing
        return True

    def find_op(self, target_op_id):
        if target_op_id in self.stat_ops:
            return self.stat_ops[target_op_id]
        elif target_op_id in self.feat_ops:
            return self.feat_ops[target_op_id]
        elif target_op_id in self.df_ops:
            return self.df_ops[target_op_id]

    def is_repeat_op(self, op, cols):
        """
        Helper function to find if a given operator targeting a column set
        already exists in the master task list.

        Parameters
        ----------
        op: operator;
        cols: str
            one of the following; continuous, categorical, all
        """
        for task_d in self.master_task_list:
            if op._id in task_d[0]._id and cols == task_d[1]:
                return True
        return False

    def run_ops_for_phase(self, gdf, tasks, record_stats=True):
        run_stat_ops = []
        for task in tasks:
            op, cols_grp, target_cols, parents = task
            LOG.debug("running op %s", op._id)
            if record_stats and op._id in self.stat_ops:
                op = self.stat_ops[op._id]
                op.apply_op(gdf, self.columns_ctx, cols_grp, target_cols=target_cols)
                run_stat_ops.append(op) if op not in run_stat_ops else None
            elif op._id in self.feat_ops:
                gdf = self.feat_ops[op._id].apply_op(
                    gdf, self.columns_ctx, cols_grp, target_cols=target_cols
                )
            elif op._id in self.df_ops:
                gdf = self.df_ops[op._id].apply_op(
                    gdf,
                    self.columns_ctx,
                    cols_grp,
                    target_cols=target_cols,
                    stats_context=self.stats,
                )
        return gdf, run_stat_ops

    # run phase
    def exec_phase(
        self,
        itr,
        phase_index,
        export_path=None,
        record_stats=True,
        shuffler=None,
        num_out_files=None,
        huge_ctr=None,
    ):
        """
        Gather necessary column statistics in single pass.
        Execute one phase only, given by phase index
        """
        LOG.debug("running phase %s", phase_index)
        stat_ops_ran = []
        for gdf in itr:
            # run all previous phases to get df to correct state
            start = time.time()
            for i in range(phase_index):
                gdf, _ = self.run_ops_for_phase(gdf, self.phases[i], record_stats=False)
            self.timings["preproc_reapply"] += time.time() - start
            start = time.time()
            gdf, stat_ops_ran = self.run_ops_for_phase(
                gdf, self.phases[phase_index], record_stats=record_stats
            )
            self.timings["preproc_apply"] += time.time() - start

            if export_path and phase_index == len(self.phases) - 1:
                self.write_df(
                    gdf, export_path, shuffler=shuffler, num_out_files=num_out_files,
                )

            if huge_ctr and phase_index == len(self.phases) - 1:
                huge_ctr.add_data(gdf)

            gdf = None
        # if export is activated combine as many GDFs as possible and
        # then write them out cudf.concat([exp_gdf, gdf], axis=0)
        for stat_op in stat_ops_ran:
            stat_op.read_fin()
            # missing bubble up to preprocessor
        self.get_stats()

    def apply(
        self,
        dataset,
        apply_offline=True,
        record_stats=True,
        shuffle=False,
        output_path="./ds_export",
        num_out_files=None,
        hugectr_gen_output=False,
        hugectr_output_path="./hugectr",
        hugectr_num_out_files=None,
    ):

        """
        Runs all the preprocessing and feature engineering operators.
        Also, shuffles the data if shuffle is set to True.

        Parameters
        -----------
        dataset : object
        apply_offline : boolean
            runs operators in offline mode or not
        record_stats : boolean
            record the stats in file or not
        shuffle : boolean
            shuffles the data or not
        output_path : string
            path to export stats
        num_out_files : integer
            number of files to create after shuffling
            the data
        """

        # if no tasks have been loaded then we need to load internal config\
        shuffler = None
        huge_ctr = None
        if not self.phases:
            self.finalize()
        if shuffle:
            shuffler = Shuffler(output_path, num_out_files=num_out_files)
        if hugectr_gen_output:
            # TODO: number of labels should be calculated here and passed to num_labels
            huge_ctr = HugeCTR(
                hugectr_output_path,
                cats=self.columns_ctx["categorical"]["base"],
                conts=self.columns_ctx["continuous"]["base"],
                labels=self.columns_ctx["label"]["base"],
                num_out_files=hugectr_num_out_files,
            )
        if apply_offline:
            self.update_stats(
                dataset,
                output_path=output_path,
                record_stats=record_stats,
                shuffler=shuffler,
                num_out_files=num_out_files,
                huge_ctr=huge_ctr,
            )
        else:
            self.apply_ops(
                dataset,
                output_path=output_path,
                record_stats=record_stats,
                shuffler=shuffler,
                num_out_files=num_out_files,
                huge_ctr=huge_ctr,
            )
        if shuffle:
            shuffler.close()
        if huge_ctr:
            huge_ctr.close()

    def update_stats(
        self,
        itr,
        end_phase=None,
        output_path=None,
        record_stats=True,
        shuffler=None,
        num_out_files=None,
        huge_ctr=None,
    ):
        end = end_phase if end_phase else len(self.phases)
        for idx, _ in enumerate(self.phases[:end]):
            self.exec_phase(
                itr,
                idx,
                export_path=output_path,
                record_stats=record_stats,
                shuffler=shuffler,
                num_out_files=num_out_files,
                huge_ctr=huge_ctr,
            )

    def apply_ops(
        self,
        gdf,
        start_phase=None,
        end_phase=None,
        record_stats=False,
        shuffler=None,
        output_path=None,
        num_out_files=None,
        huge_ctr=None,
    ):
        """
        gdf: cudf dataframe
        record_stats: bool; run stats recording within run
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
            gdf, stat_ops_ran = self.run_ops_for_phase(
                gdf, self.phases[phase_index], record_stats=record_stats
            )
            self.timings["preproc_apply"] += time.time() - start
            if phase_index == len(self.phases) - 1 and output_path:
                self.write_df(
                    gdf, output_path, shuffler=shuffler, num_out_files=num_out_files,
                )
        return gdf

    @annotate("Write_df", color="red", domain="nvt_python")
    def write_df(self, gdf, export_path, shuffler, num_out_files):
        if shuffler:
            start = time.time()
            shuffler.add_data(gdf)
            self.timings["shuffle_df"] += time.time() - start
        else:
            file_name = f"{self.current_file_num}.parquet"
            path = os.path.join(export_path, file_name)
            gdf.to_parquet(path, compression=None)
            self.current_file_num += 1

    def get_stats(self):
        for name, stat_op in self.stat_ops.items():
            stat_vals = stat_op.stats_collected()
            for name, stat in stat_vals:
                if name in self.stats:
                    self.stats[name] = stat
                else:
                    warnings.warn("stat not found,", name)

    def save_stats(self, path):
        main_obj = {}
        stats_drop = {}
        stats_drop["encoders"] = {}
        encoders = self.stats.get("encoders", {})
        for name, enc in encoders.items():
            stats_drop["encoders"][name] = (enc.get_cats().values_to_string(),)
        for name, stat in self.stats.items():
            if name not in stats_drop.keys():
                stats_drop[name] = stat
        main_obj["stats"] = stats_drop
        main_obj["columns_ctx"] = self.columns_ctx
        op_args = {}
        tasks = []
        for task in self.master_task_list:
            tasks.append([task[0]._id, task[1], task[2], [x._id for x in task[3]]])
            op = self.find_op(task[0]._id)
            op_args[op._id] = op.__dict__
        main_obj["op_args"] = op_args
        main_obj["tasks"] = tasks
        with open(path, "w") as outfile:
            yaml.dump(main_obj, outfile, default_flow_style=False)

    def load_stats(self, path):
        def _set_stats(self, stats_dict):
            for key, stat in stats_dict.items():
                self.stats[key] = stat

        with open(path, "r") as infile:
            main_obj = yaml.load(infile)
            _set_stats(self, main_obj["stats"])
            self.master_task_list = self.recreate_master_task_list(
                main_obj["tasks"], main_obj["op_args"]
            )
            self.columns_ctx = main_obj["columns_ctx"]
        encoders = self.stats.get("encoders", {})
        for col, cats in encoders.items():
            self.stats["encoders"][col] = DLLabelEncoder(col, cats=cudf.Series(cats[0]))
        self.reg_all_ops(self.master_task_list)

    def clear_stats(self):

        for stat, vals in self.stats.items():
            self.stats[stat] = {}

        for statop_id, stat_op in self.stat_ops.items():
            stat_op.clear()

    def ds_to_tensors(self, itr, apply_ops=True):
        from nvtabular.torch_dataloader import create_tensors

        return create_tensors(self, itr=itr, apply_ops=apply_ops)

    def recreate_master_task_list(self, task_list, op_args):
        master_list = []
        for task in task_list:
            op_id = task[0]
            main_grp = task[1]
            sub_cols = task[2]
            dep_ids = task[3]
            op = all_ops[op_id](**op_args[op_id])
            dep_ops = []
            for ops_id in dep_ids:
                dep_ops.append(all_ops[ops_id]())

            master_list.append((op, main_grp, sub_cols, dep_ops))
        return master_list


def get_new_config():
    """
    boiler config object, to be filled in with targeted operator tasks
    """
    config = {}
    config["FE"] = {}
    config["FE"]["all"] = {}
    config["FE"]["continuous"] = {}
    config["FE"]["categorical"] = {}
    config["PP"] = {}
    config["PP"]["all"] = {}
    config["PP"]["continuous"] = {}
    config["PP"]["categorical"] = {}
    return config


def get_new_list_config():
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


def _shuffle_part(gdf):
    sort_key = "__sort_index__"
    arr = cp.arange(len(gdf))
    cp.random.shuffle(arr)
    gdf[sort_key] = cudf.Series(arr)
    return gdf.sort_values(sort_key).drop(columns=[sort_key])
