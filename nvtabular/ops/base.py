class OperatorBlock:
    def __init__(self, *ops, auto_renaming=False, sequential=True):
        self._ops = list(ops) if ops else []
        self._ops_by_name = {}
        self.sequential = sequential
        self.auto_renaming = auto_renaming

    def add(self, op, name=None):
        self._ops.append(op)
        if name:
            self._ops_by_name[name] = op

        return op

    def extend(self, ops):
        self._ops.extend(ops)

        return ops

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._ops[key]

        return self._ops_by_name[key]

    @property
    def ops(self):
        return self._ops

    def __rrshift__(self, other):
        from nvtabular.column_group import ColumnGroup

        return self(ColumnGroup(other))

    def __call__(self, col_or_cols, add=False):
        from nvtabular.ops import Operator

        x = col_or_cols
        name_parts = []

        if self.sequential:
            for op in self._ops:
                if isinstance(op, Operator):
                    name_parts.append(op.__class__.__name__)
                    x = x >> op
                else:
                    x = op(x)
        else:
            out = None
            for op in self._ops:
                if out:
                    out += col_or_cols >> op
                else:
                    out = col_or_cols >> op
            x = out

        if self.auto_renaming:
            from nvtabular.ops import Rename

            x = x >> Rename(postfix="/" + "/".join(name_parts))

        if add:
            return col_or_cols + x

        return x

    def copy(self) -> "OperatorBlock":
        to_return = OperatorBlock(
            *self._ops, auto_renaming=self.auto_renaming, sequential=self.sequential
        )
        to_return._ops_by_name = self._ops_by_name

        return to_return
