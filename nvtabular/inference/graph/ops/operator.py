from abc import abstractclassmethod, abstractmethod

import nvtabular as nvt
from nvtabular.graph.base_operator import BaseOperator


class InferenceDataFrame:
    def __init__(self, tensors=None):
        self.tensors = tensors or {}

    def __getitem__(self, col_items):
        if isinstance(col_items, list):
            results = {name: self.tensors[name] for name in col_items}
            return InferenceDataFrame(results)
        else:
            return self.tensors[col_items]

    def __len__(self):
        return len(self.tensors)

    def __iter__(self):
        for tensor in self.tensors.items():
            yield tensor

    def __repr__(self):
        dict_rep = {}
        for k, v in self.tensors.items():
            dict_rep[k] = v
        return str(dict_rep)


class InferenceOperator(BaseOperator):
    @property
    @abstractmethod
    def export_name(self):
        pass

    @abstractmethod
    def export(self, path):
        pass

    def create_node(self, selector):
        return nvt.inference.graph.node.InferenceNode(selector)


class PipelineableInferenceOperator(InferenceOperator):
    @abstractclassmethod
    def from_config(cls, config):
        pass

    @abstractmethod
    def transform(self, df: InferenceDataFrame) -> InferenceDataFrame:
        """Transform the dataframe by applying this operator to the set of input columns

        Parameters
        -----------
        df: Dataframe
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        DataFrame
            Returns a transformed dataframe for this operator
        """
