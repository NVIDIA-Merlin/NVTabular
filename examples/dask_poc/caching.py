from io import BytesIO
import cudf
from cudf.io.parquet import ParquetWriter
from pyarrow.compat import guid
import pandas as pd


class CategoryCache:
    def __init__(self):
        self.cat_cache = {}
        self.pq_writer_cache = {}

    def __del__(self):
        for path, (pw, fpath) in self.pq_writer_cache.items():
            pw.close()

    def _get_pq_writer(self, prefix, s, mem):
        pw, fil = self.pq_writer_cache.get(prefix, (None, None))
        if pw is None:
            if mem:
                fil = BytesIO()
                pw = ParquetWriter(fil, compression=None)
                self.pq_writer_cache[prefix] = (pw, fil)
            else:
                outfile_id = guid() + ".parquet"
                full_path = ".".join([prefix, outfile_id])
                pw = ParquetWriter(full_path, compression=None)
                self.pq_writer_cache[prefix] = (pw, full_path)
        return pw

    def _get_encodings(self, col, path, cache=False):
        table = self.cat_cache.get(col, None)
        if table and not isinstance(table, cudf.DataFrame):
            df = cudf.io.read_parquet(table, index=False, columns=[col])
            df.index.name = "labels"
            df.reset_index(drop=False, inplace=True)
            return df

        if table is None:
            if cache:
                table = cudf.io.read_parquet(path, index=False, columns=[col])
            else:
                with open(path, "rb") as f:
                    self.cat_cache[col] = BytesIO(f.read())
                table = cudf.io.read_parquet(self.cat_cache[col], index=False, columns=[col])
            table.index.name = "labels"
            table.reset_index(drop=False, inplace=True)
            if cache:
                self.cat_cache[col] = table.copy(deep=False)

        return table


cats_cache = None


def cache():
    global cats_cache
    if cats_cache is None:
        cats_cache = CategoryCache()
    return cats_cache


def clean():
    global cats_cache
    if cats_cache is not None:
        del cats_cache
    return
