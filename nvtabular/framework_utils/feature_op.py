from typing import Optional, List, Union

from nvtabular.feature_group import FeatureGroup

ColumnNames = List[Union[str, List[str]]]


class FeatureGroupOp(object):
    def call(self, columns: ColumnNames, inputs, **kwargs):
        raise NotImplementedError

    def filter_columns(self, columns: ColumnNames, inputs):
        return {k: v for k, v in inputs.items() if k in columns}

    def output_column_names(self, columns: ColumnNames) -> ColumnNames:
        return columns

    def dependencies(self) -> Optional[List[Union[str, FeatureGroup]]]:
        """Defines an optional list of column dependencies for this operator. This lets you consume columns
        that aren't part of the main transformation workflow.

        Returns
        -------
        str, list of str or ColumnGroup, optional
            Extra dependencies of this operator. Defaults to None
        """
        return None

    def __rrshift__(self, other) -> FeatureGroup:
        return FeatureGroup(other) >> self

    @property
    def label(self) -> str:
        return self.__class__.__name__
