from abc import abstractmethod

from nvtabular.graph.base_operator import BaseOperator


class InferenceOperator(BaseOperator):
    @property
    @abstractmethod
    def export_name(self):
        pass

    @abstractmethod
    def export(self, path):
        pass
