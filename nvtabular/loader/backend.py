#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import math
import queue
import threading
import warnings
from collections import OrderedDict

import cupy as cp

from nvtabular.dispatch import _concat, _is_list_dtype, _make_df, _pull_apart_list
from nvtabular.io.shuffle import _shuffle_df
from nvtabular.ops import _get_embedding_order


def _num_steps(num_samples, step_size):
    return math.ceil(num_samples / step_size)


class ChunkQueue:
    """This class takes partitions (parts) from an NVTabular dataset
     and concatenates them into a cudf dataframe "chunk". This chunk
    is subsequently transformed into its tensor representation using
    the iterator's transform.

    Parameters
    -----------
    qsize: int
        Max number of elements to hold in the buffer at once
    num_parts : int
        number of partitions from the iterator, an NVTabular Dataset to concatenate into a "chunk"
    shuffle : bool
        enable/disable chunk-level shuffling
    put_wait: float
        amount of timeout to wait for a full queue to open up
        before checking for errors and trying again
    """

    def __init__(self, dataloader, qsize, num_parts=1, shuffle=False, put_wait=1e-6):
        self.num_parts = num_parts
        self.shuffle = shuffle
        self.put_wait = put_wait
        self.q_out = queue.Queue(qsize)
        self._stop_event = threading.Event()
        indices = dataloader._gather_indices_for_dev(0)
        self.itr = dataloader.data.to_iter(indices=indices)
        self.dataloader = dataloader

    def __len__(self):
        return len(self.itr)

    @property
    def stopped(self):
        return self._stop_event.is_set()

    @property
    def empty(self):
        return self.q_out.empty()

    def get(self):
        return self.q_out.get()

    def put(self, packet):
        while True:
            if self.stopped:
                return True

            try:
                self.q_out.put(packet, timeout=self.put_wait)
                return False
            except queue.Full:
                continue

    def batch(self, itr):
        """
        iterates through gpu_mem_frac size chunks of dataset
        and concatenates every `num_parts` of them.
        """
        current = []
        while True:
            try:
                value = next(itr)
            except StopIteration:
                if len(current) > 0:
                    yield current
                break

            current.append(value)
            if len(current) == self.num_parts:
                yield current
                current = []

    def chunk_logic(self, itr):
        spill = None
        for chunks in self.batch(itr):
            if self.stopped:
                return

            if spill and not spill.empty:
                chunks.insert(0, spill)

            chunks = _concat(chunks)
            chunks.reset_index(drop=True, inplace=True)
            chunks, spill = self.get_batch_div_chunk(chunks, self.dataloader.batch_size)
            if self.shuffle:
                chunks = _shuffle_df(chunks)

            if len(chunks) > 0:
                chunks = self.dataloader.make_tensors(chunks, self.dataloader._use_nnz)
                # put returns True if buffer is stopped before
                # packet can be put in queue. Keeps us from
                # freezing on a put on a full queue
                if self.put(chunks):
                    return
            chunks = None
        # takes care final batch, which is less than batch size
        if not self.dataloader.drop_last and spill is not None and not spill.empty:
            spill = self.dataloader.make_tensors(spill, self.dataloader._use_nnz)
            self.put(spill)

    def load_chunks(self, dev):
        try:
            itr = iter(self.itr)
            if self.dataloader.device != "cpu":
                with self.dataloader._get_device_ctx(dev):
                    self.chunk_logic(itr)
            else:
                self.chunk_logic(itr)
        except Exception as e:  # pylint: disable=broad-except
            self.put(e)

    # For when an iterator is stopped before iteration is complete.
    def stop(self):
        self._stop_event.set()
        # TODO: should we be clearing? I can imagine a world where
        # you want the thread to stop but still want to grab
        # data out of the buffer
        self.q_out.queue.clear()

    def start(self):
        self._stop_event.clear()

    def get_batch_div_chunk(self, chunks, batch_size):
        # TODO: is there a way to do this using cupy?
        spill_idx = int(chunks.shape[0] / batch_size) * batch_size
        spill = _make_df(chunks.iloc[spill_idx:])
        chunks = _make_df(chunks.iloc[:spill_idx])
        if not chunks.empty:
            chunks.reset_index(drop=True, inplace=True)
        if not spill.empty:
            spill.reset_index(drop=True, inplace=True)
        return chunks, spill


# TODO: implement as metaclass and assign methods to children
# to avoid having to do Dataset.<method> calls?
class DataLoader:
    _use_nnz = False

    def __init__(
        self,
        dataset,
        cat_names,
        cont_names,
        label_names,
        batch_size,
        shuffle,
        seed_fn=None,
        parts_per_chunk=1,
        device=None,
        global_size=None,
        global_rank=None,
        drop_last=False,
        sparse_names=None,
        sparse_max=None,
        sparse_as_dense=False,
    ):
        self.data = dataset
        self.indices = cp.arange(dataset.to_ddf().npartitions)
        self.drop_last = drop_last
        self.device = device or 0
        self.sparse_names = sparse_names or []
        self.sparse_max = sparse_max or {}
        self.sparse_as_dense = sparse_as_dense
        self.global_size = global_size or 1
        self.global_rank = global_rank or 0

        self.cat_names = cat_names or []
        self.cont_names = cont_names or []
        self.label_names = label_names
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed_fn = seed_fn

        self.num_rows_processed = 0

        # we set size of chunk queue to 1 we only want one chunk in queue at a time.
        self._buff = ChunkQueue(self, 1, num_parts=parts_per_chunk, shuffle=shuffle)
        # run once instead of everytime len called
        self._buff_len = len(self._buff)
        self._batch_itr = None
        self._workers = None

    def __len__(self):
        batches = _num_steps(self._buff_len, self.batch_size)
        if self.drop_last and self._buff_len % self.batch_size > 0:
            batches = batches - 1
        return batches

    @property
    def _working(self):
        if self._workers is not None:
            return any(t.is_alive() for t in self._workers)
        return False

    def stop(self):
        # TODO: raise warning or even error if condition
        # isn't met?
        if self._workers is not None:
            if not self._buff.stopped:
                self._buff.stop()
            for t in self._workers:
                t.join()
            # remove joined threads from list
            self._workers = None
            self._buff.q_out.queue.clear()
        self._batch_itr = None

    def _gather_indices_for_dev(self, dev):
        # this should be self.indices divided by total processes, global set
        if len(self.indices) < self.global_size:
            warnings.warn(
                f"""You have more processes({self.global_size}) than dataset
                    partitions({len(self.indices)}), reduce the number of processes."""
            )
            raise IndexError
        per_worker = _num_steps(len(self.indices), self.global_size)
        # identify process rank out of all processes (not local rank)
        start = self.global_rank * per_worker
        return self.indices[start : start + per_worker].tolist()

    def _generate_local_seed(self):
        random_state = cp.random.get_random_state()
        seeds = random_state.tomaxint(size=self.global_size)
        local_seed = seeds[self.global_rank]
        cp.random.seed(local_seed.get())

    def _shuffle_indices(self):
        self._generate_local_seed()
        if self.seed_fn:
            new_seed = self.seed_fn()
            cp.random.seed(new_seed)
        cp.random.shuffle(self.indices)
        self._generate_local_seed()

    def __iter__(self):
        self.stop()
        self.num_rows_processed = 0
        if self._buff.stopped:
            self._buff.start()

        # shuffle partition indices to bring disparate
        # parts of the dataset "close" to one another
        if self.shuffle:
            self._shuffle_indices()

        # build and start new threads for loading and
        # concatenating data
        self._workers = []
        t = threading.Thread(target=self._buff.load_chunks, args=(self.device,))
        t.daemon = True
        t.start()
        self._workers.append(t)
        return self

    def __next__(self):
        return self._get_next_batch()

    def _fetch_chunk(self):
        chunks = self._buff.get()
        if isinstance(chunks, Exception):
            self.stop()
            raise chunks
        self._batch_itr = iter(chunks)

    def _get_next_batch(self):
        """
        adding this cheap shim so that we can call this
        step without it getting overridden by the
        framework-specific parent class's `__next__` method.
        TODO: can this be better solved with a metaclass
        implementation? My gut is that we don't actually
        necessarily *want*, in general, to be overriding
        __next__ and __iter__ methods
        """
        # we've never initialized, do that now
        # need this because tf.keras.Model.fit will
        # call next() cold
        if self._workers is None:
            DataLoader.__iter__(self)

        # get the first chunks
        if self._batch_itr is None:
            self._fetch_chunk()

        # try to iterate through existing batches
        try:
            batch = next(self._batch_itr)
        except StopIteration:
            # anticipate any more chunks getting created
            # if not, raise the StopIteration
            if not self._working and self._buff.empty:
                self._workers = None
                self._batch_itr = None
                raise

            # otherwise get the next chunks and return
            # the first batch
            self._fetch_chunk()
            batch = next(self._batch_itr)
        # if batch[0] is empty but other exist
        for sub in batch:
            if sub is not None and len(sub) > 0:
                self.num_rows_processed += len(sub)
                break
        return batch

    def make_tensors(self, gdf, use_nnz=False):
        split_idx = self._get_segment_lengths(len(gdf))

        # map from big chunk to framework-specific tensors
        chunks = self._create_tensors(gdf)

        # if we have any offsets, calculate nnzs up front
        if len(chunks) == 4:
            offsets = chunks[-1]
            if use_nnz:
                nnzs = offsets[1:] - offsets[:-1]
            chunks = chunks[:-1]

        # split them into batches and map to the framework-specific output format
        batches = [[] for _ in range(len(split_idx))]
        offset_idx = 0
        for chunk in chunks:
            lists = None
            if isinstance(chunk, tuple):
                chunk, lists = chunk

            if len(split_idx) > 1 and chunk is not None:
                chunk = self._split_fn(chunk, split_idx)
            else:
                chunk = [chunk for _ in split_idx]

            if lists is not None:
                num_list_columns = len(lists)

                # grab the set of offsets and nnzs corresponding to
                # the list columns from this chunk
                chunk_offsets = offsets[:, offset_idx : offset_idx + num_list_columns]
                if use_nnz:
                    chunk_nnzs = nnzs[:, offset_idx : offset_idx + num_list_columns]
                offset_idx += num_list_columns

                # split them into batches, including an extra 1 on the offsets
                # so we know how long the very last element is
                batch_offsets = self._split_fn(chunk_offsets, split_idx + [1])
                if use_nnz and len(split_idx) > 1:
                    batch_nnzs = self._split_fn(chunk_nnzs, split_idx)
                elif use_nnz:
                    batch_nnzs = [chunk_nnzs]
                else:
                    batch_nnzs = [None] * (len(batch_offsets) - 1)

                # group all these indices together and iterate through
                # them in batches to grab the proper elements from each
                # values tensor
                chunk = zip(chunk, batch_offsets[:-1], batch_offsets[1:], batch_nnzs)

            for n, c in enumerate(chunk):
                if isinstance(c, tuple):
                    c, off0s, off1s, _nnzs = c
                    offsets_split_idx = [1 for _ in range(num_list_columns)]
                    off0s = self._split_fn(off0s, offsets_split_idx, axis=1)
                    off1s = self._split_fn(off1s, offsets_split_idx, axis=1)
                    if use_nnz:
                        _nnzs = self._split_fn(_nnzs, offsets_split_idx, axis=1)

                    # TODO: does this need to be ordereddict?
                    batch_lists = {}
                    for k, (column_name, values) in enumerate(lists.items()):
                        off0, off1 = off0s[k], off1s[k]
                        if use_nnz:
                            nnz = _nnzs[k]

                        # need to grab scalars for TF case
                        if len(off0.shape) == 1:
                            start, stop = off0[0], off1[0]
                        elif len(off0.shape) == 2:
                            start, stop = off0[0, 0], off1[0, 0]
                        else:
                            print(off0, off1)
                            raise ValueError

                        value = values[start:stop]
                        index = off0 - start if not use_nnz else nnz
                        batch_lists[column_name] = (value, index)
                    c = (c, batch_lists)

                batches[n].append(c)
        return [self._handle_tensors(*batch) for batch in batches]

    def _get_segment_lengths(self, num_samples):
        """
        Helper function to build indices to pass
        to <torch|tf>.split functions for breaking
        up into batches
        """
        num_full_batches = _num_steps(num_samples, self.batch_size) - 1
        idx = [self.batch_size for _ in range(num_full_batches)]
        idx.append(num_samples - num_full_batches * self.batch_size)
        return idx

    def _to_sparse_tensor(self, values_offset, column_name):
        """
        Create a sparse representation of the input tensor.
        values_offset is either a tensor or a tuple of tensor, offset.
        """
        seq_limit = self.sparse_max[column_name]
        values, offsets, diff_offsets, num_rows = self._pull_values_offsets(values_offset)
        max_seq_len = self._get_max_seq_len(diff_offsets)
        if max_seq_len > seq_limit:
            raise ValueError(
                "The default sequence length has been configured "
                + f"to {seq_limit} but the "
                + f"largest sequence in this batch have {max_seq_len} length"
            )
        return self._build_sparse_tensor(values, offsets, diff_offsets, num_rows, seq_limit)

    def _to_tensor(self, gdf, dtype=None):
        """
        One of the mandatory functions a child class needs
        to implement. Maps from a cudf DataFrame to a
        tensor in the appropriate library, with an optional
        dtype kwarg to do explicit casting if need be
        """
        raise NotImplementedError

    def _get_device_ctx(self, dev):
        """
        One of the mandatory functions a child class needs
        to implement. Maps from a GPU index to a framework
        context object for placing tensors on specific GPUs
        """
        raise NotImplementedError

    def _split_fn(self, tensor, idx, axis=0):
        raise NotImplementedError

    @property
    def _LONG_DTYPE(self):
        raise NotImplementedError

    @property
    def _FLOAT32_DTYPE(self):
        raise NotImplementedError

    def _separate_list_columns(self, gdf):
        lists, scalars = [], []
        for col in gdf.columns:
            if _is_list_dtype(gdf[col]):
                lists.append(col)
            else:
                scalars.append(col)
        return _get_embedding_order(scalars), _get_embedding_order(lists)

    def _create_tensors(self, gdf):
        """
        Breaks a dataframe down into the relevant
        categorical, continuous, and label tensors.
        Can be overrideen
        """
        column_groups = (self.cat_names, self.cont_names, self.label_names)
        dtypes = (self._LONG_DTYPE, self._FLOAT32_DTYPE, self._FLOAT32_DTYPE)
        tensors = []
        offsets = _make_df(device=self.device)
        for column_names, dtype in zip(column_groups, dtypes):
            if len(column_names) == 0:
                tensors.append(None)
                continue

            gdf_i = gdf[column_names]
            gdf.drop(columns=column_names, inplace=True)

            scalars, lists = self._separate_list_columns(gdf_i)

            x = None
            if scalars:
                # should always return dict column_name: values, offsets (optional)
                x = self._to_tensor(gdf_i[scalars], dtype)
            if lists:
                list_tensors = OrderedDict()
                for column_name in lists:
                    column = gdf_i.pop(column_name)
                    leaves, offsets[column_name] = _pull_apart_list(column)
                    list_tensors[column_name] = self._to_tensor(leaves, dtype)
                x = x, list_tensors
            tensors.append(x)

        if not offsets.empty:
            offsets_tensor = self._to_tensor(offsets, self._LONG_DTYPE)
            if len(offsets_tensor.shape) == 1:
                offsets_tensor = offsets_tensor[:, None]
            tensors.append(offsets_tensor)
        del gdf, offsets

        return tensors

    def _handle_tensors(self, cats, conts, labels):
        X = {}
        for tensor, names in zip([cats, conts], [self.cat_names, self.cont_names]):
            lists = {}
            if isinstance(tensor, tuple):
                tensor, lists = tensor
            names = [i for i in names if i not in lists]

            # now add in any scalar tensors
            if len(names) > 1:
                tensors = self._tensor_split(tensor, len(names), axis=1)
                lists.update(zip(names, tensors))
            elif len(names) == 1:
                lists[names[0]] = tensor
            X.update(lists)

        for column_name in X:
            if column_name in self.sparse_names:
                if column_name not in self.sparse_max:
                    raise ValueError(
                        f"Did not convert {column_name} to sparse due to missing sparse_max entry"
                    )
                X[column_name] = self._to_sparse_tensor(X[column_name], column_name)

        # TODO: use dict for labels as well?
        # would require output layers to match naming
        if len(self.label_names) > 1:
            labels = self._tensor_split(labels, len(self.label_names), axis=1)
        return X, labels
