import logging

from nvtabular import ops

LOG = logging.getLogger("nvtabular")


def remove_consecutive_interactions(
    df, session_id_col="session_id", item_id_col="item_id", timestamp_col="timestamp"
):
    LOG.info("Count with in-session repeated interactions: {}".format(len(df)))
    # Sorts the dataframe by session and timestamp, to remove consective repetitions
    df = df.sort_values([session_id_col, timestamp_col])

    # Keeping only no consectutive repeated in session interactions
    session_is_last_session = df[session_id_col] == df[session_id_col].shift(1)
    item_is_last_item = df[item_id_col] == df[item_id_col].shift(1)
    df = df[~(session_is_last_session & item_is_last_item)]
    LOG.info("Count after removed in-session repeated interactions: {}".format(len(df)))

    return df


def create_session_groupby_aggs(column_group, default_agg="list", extra_aggs=None, to_ignore=None):
    if not extra_aggs:
        extra_aggs = {}
    if not to_ignore:
        to_ignore = []

    aggs = {col: [default_agg] for col in column_group.columns if col not in to_ignore}
    for key, val in extra_aggs.items():
        if key in aggs:
            if isinstance(val, list):
                aggs[key].extend(val)
            else:
                aggs[key].append(val)
        else:
            aggs[key] = val

    return aggs


class SessionDay(ops.OperatorBlock):
    def __init__(self, name="day_idx", padding=4):
        super().__init__()
        self.extend(
            [
                ops.LambdaOp(lambda x: (x - x.min()).dt.days + 1),
                ops.LambdaOp(lambda col: col.astype(str).str.pad(padding, fillchar="0")),
                ops.Rename(f=lambda col: name),
            ]
        )
