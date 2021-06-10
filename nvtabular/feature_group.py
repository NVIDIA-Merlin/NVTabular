import abc

from nvtabular.column_group import ColumnGroup
from nvtabular.workflow import Workflow


class FeatureGroup(ColumnGroup):
    @classmethod
    def from_column_group(cls, column_group: ColumnGroup):
        return cls(column_group.columns)

    @classmethod
    def from_workflow(cls, workflow: Workflow):
        return cls(workflow.column_group.columns)

    @classmethod
    def from_dtype_in_workflow(cls, workflow: Workflow, dtype, is_list=False):
        filtered = []
        for name, f_type in workflow.output_dtypes.items():
            if not is_list and f_type == dtype:
                filtered.append(name)
            elif is_list and f_type.leaf_type == dtype:
                filtered.append(name)

        return cls(filtered)

    @abc.abstractmethod
    def __call__(self, operator, **kwargs):
        raise NotImplementedError


class TargetGroup(ColumnGroup):
    @classmethod
    def from_column_group(cls, column_group: ColumnGroup):
        return cls(column_group.columns)

    @classmethod
    def from_workflow(cls, workflow: Workflow):
        return cls(workflow.column_group.columns)

    @abc.abstractmethod
    def __call__(self, operator, **kwargs):
        raise NotImplementedError