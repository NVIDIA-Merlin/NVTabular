import numpy as np

from nvtabular.ops import Normalize
from nvtabular.ops.base import AddMetadata, OperatorBlock
from nvtabular.ops.lambdaop import LambdaOp
from nvtabular.ops.rename import Rename
from nvtabular.tag import DefaultTags


class TimestampFeatures(OperatorBlock):
    def __init__(
        self,
        add_date_parts=True,
        normalize_date_parts=True,
        add_timestamp=True,
        add_cycled=True,
        auto_renaming=False,
        delimiter="/",
    ):
        super().__init__(auto_renaming=auto_renaming, sequential=False)
        self.normalize_date_parts = normalize_date_parts
        self.delimiter = delimiter
        if add_cycled or add_date_parts:
            hour = self.add(
                OperatorBlock(
                    LambdaOp(lambda col: col.dt.hour),
                    *self._rename_and_tag("hour", normalize=normalize_date_parts),
                )
            )
            weekday = self.add(
                OperatorBlock(
                    LambdaOp(lambda col: col.dt.weekday),
                    *self._rename_and_tag("weekday", normalize=normalize_date_parts),
                )
            )
            self.add(
                OperatorBlock(
                    LambdaOp(lambda col: col.dt.day),
                    *self._rename_and_tag("day", normalize=normalize_date_parts),
                )
            )
            self.add(
                OperatorBlock(
                    LambdaOp(lambda col: col.dt.month),
                    *self._rename_and_tag("month", normalize=normalize_date_parts),
                )
            )
            self.add(
                OperatorBlock(
                    LambdaOp(lambda col: col.dt.year),
                    *self._rename_and_tag("year", normalize=normalize_date_parts),
                )
            )

            if add_cycled:
                self.add(
                    OperatorBlock(
                        *hour.ops,
                        LambdaOp(lambda col: self.get_cycled_feature_value_sin(col, 24)),
                        *self._rename_and_tag("sin"),
                    )
                )
                self.add(
                    OperatorBlock(
                        *hour.ops,
                        LambdaOp(lambda col: self.get_cycled_feature_value_cos(col, 24)),
                        *self._rename_and_tag("cos"),
                    )
                )
                self.add(
                    OperatorBlock(
                        *weekday.ops,
                        LambdaOp(lambda col: self.get_cycled_feature_value_sin(col + 1, 7)),
                        *self._rename_and_tag("sin"),
                    )
                )
                self.add(
                    OperatorBlock(
                        *weekday.ops,
                        LambdaOp(lambda col: self.get_cycled_feature_value_cos(col + 1, 7)),
                        *self._rename_and_tag("cos"),
                    )
                )

        if add_timestamp:
            self.add(
                OperatorBlock(
                    LambdaOp(lambda col: (col.astype(int) / 1e6).astype(int)),
                    Rename(f=lambda col: "ts"),
                    AddMetadata(tags=DefaultTags.TIME),
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

    def _rename_and_tag(self, postfix, normalize=False):
        outputs = [Rename(postfix=f"{self.delimiter}{postfix}"), AddMetadata(tags=DefaultTags.TIME)]

        if normalize:
            outputs.append(Normalize())

        return outputs
