import pytest

inf_op = pytest.importorskip("nvtabular.inference.graph.ops.operator")


class PlusTwoOp(inf_op.PipelineableInferenceOperator):
    def transform(self, df: inf_op.InferenceDataFrame) -> inf_op.InferenceDataFrame:
        focus_df = df
        new_df = inf_op.InferenceDataFrame()

        for name, data in focus_df:
            new_df.tensors[f"{name}_plus_2"] = data + 2

        return new_df

    def column_mapping(self, col_selector):
        column_mapping = {}
        for col_name in col_selector.names:
            column_mapping[f"{col_name}_plus_2"] = [col_name]
        return column_mapping

    @classmethod
    def from_config(cls, config):
        return PlusTwoOp()
