import numpy as np

from nvtabular.ops.base import OperatorBlock
from nvtabular.ops.lambdaop import LambdaOp
from nvtabular.ops.rename import Rename


class TimestampFeatures(OperatorBlock):
    def __init__(self, add_timestamp=True, add_cycled=True, auto_renaming=False, delimiter="/"):
        super().__init__(auto_renaming=auto_renaming, sequential=False)
        del_fn = lambda x: f"{delimiter}{x}"
        hour = self.add(
            OperatorBlock(LambdaOp(lambda col: col.dt.hour), Rename(postfix=del_fn("hour")))
        )
        weekday = self.add(
            OperatorBlock(
                LambdaOp(lambda col: col.dt.weekday),
                Rename(postfix=del_fn("weekday")),
            )
        )
        self.add(OperatorBlock(LambdaOp(lambda col: col.dt.day), Rename(postfix=del_fn("day"))))
        self.add(
            OperatorBlock(
                LambdaOp(lambda col: col.dt.month),
                Rename(postfix=del_fn("month")),
            )
        )
        self.add(OperatorBlock(LambdaOp(lambda col: col.dt.year), Rename(postfix=del_fn("year"))))

        if add_timestamp:
            self.add(
                OperatorBlock(
                    LambdaOp(lambda col: (col.astype(int) / 1e6).astype(int)),
                    Rename(f=lambda col: "ts"),
                )
            )

        if add_cycled:
            self.add(
                OperatorBlock(
                    *hour.ops,
                    LambdaOp(lambda col: self.get_cycled_feature_value_sin(col, 24)),
                    Rename(postfix="_sin"),
                )
            )
            self.add(
                OperatorBlock(
                    *hour.ops,
                    LambdaOp(lambda col: self.get_cycled_feature_value_cos(col, 24)),
                    Rename(postfix="_cos"),
                )
            )
            self.add(
                OperatorBlock(
                    *weekday.ops,
                    LambdaOp(lambda col: self.get_cycled_feature_value_sin(col + 1, 7)),
                    Rename(postfix="_sin"),
                )
            )
            self.add(
                OperatorBlock(
                    *weekday.ops,
                    LambdaOp(lambda col: self.get_cycled_feature_value_cos(col + 1, 7)),
                    Rename(postfix="_cos"),
                )
            )

    @staticmethod
    def get_cycled_feature_value_sin(col, max_value):
        value_scaled = (col + 0.000001) / max_value
        value_sin = np.sin(2 * np.pi * value_scaled)

        return value_sin

    @staticmethod
    def get_cycled_feature_value_cos(col, max_value):
        value_scaled = (col + 0.000001) / max_value
        value_cos = np.cos(2 * np.pi * value_scaled)

        return value_cos
