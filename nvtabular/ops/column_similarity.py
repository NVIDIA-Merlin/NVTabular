#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
import cudf
import cupy
import cupy.sparse
import numba
import numpy
import scipy.sparse
from nvtx import annotate

from .operator import CAT, CONT
from .transform_operator import TransformOperator


class ColumnSimilarity(TransformOperator):
    """Calculates the similarity between two columns using tf-idf, cosine or
    inner product as the distance metric. For each row, this calculates the distance
    between the two columns by looking up features for those columns in a sparse matrix,
    and then computing the distance between the rows of the feature matrices.

    Example usage::

        # Read in the 'document_categories' file from the kaggle outbrains dataset and convert
        # to a sparse matrix
        df = cudf.read_csv("document_categories.csv.zip")
        categories = cupy.sparse.coo_matrix((cupy.ones(len(df)),
                                            (df.document_id.values, df.category_id.values))
        # compute a new column 'similarity' between the document_id and promo_document_id columns
        # on tfidf distance on the categories matrix we just loaded up
        workflow.add_feature(ColumnSimilarity("similarity", "document_id", categories,
                                              "promo_document_id"))

    Parameters
    -----------
    name : str
        Name of the output column
    a : str
        Name of the first column to calculate similarity for
    a_features : csr_matrix
        Sparse feature matrix for the 'a' column
    b : str
        Name of the second column to calculate similarity for
    b_features : csr_matrix, optional
        Sparse feature matrix for the 'b' column. If not given will use the
        same feature matrix as for 'a' (for example when calculating document-document distances)
    on_device : bool
        Whether to compute on the GPU or CPU. Computing on the GPU will be
        faster, but requires that the a_features/b_features sparse matrices
        fit into GPU memory.
    """

    default_in = CAT
    default_out = CONT

    def __init__(
        self, name, a_col, a_features, b_col, b_features=None, metric="tfidf", on_device=True
    ):
        super(ColumnSimilarity, self).__init__(columns=[a_col, b_col], replace=False)
        self.name = name
        self.a_col = a_col
        self.b_col = b_col

        self.a_features = _convert_features(a_features, metric, on_device)
        self.b_features = (
            _convert_features(b_features, metric, on_device)
            if b_features is not None
            else self.a_features
        )
        self.on_device = on_device

    @annotate("ColumnSimilarity_op", color="darkgreen", domain="nvt_python")
    def apply_op(
        self,
        gdf: cudf.DataFrame,
        columns_ctx: dict,
        input_cols,
        target_cols=["base"],
        stats_context=None,
    ):
        a = gdf[self.a_col].values if self.on_device else gdf[self.a_col].values_host
        b = gdf[self.b_col].values if self.on_device else gdf[self.b_col].values_host

        if len(a) and len(b):
            similarities = row_wise_inner_product(
                a, self.a_features, b, self.b_features, self.on_device
            )
        else:
            similarities = []
        gdf[self.name] = similarities

        columns_ctx[input_cols][self._id] = [self.name]
        return gdf

    @property
    def _id(self):
        return f"{self.__class__.__name__}_{self.name}"


def row_wise_inner_product(a, a_features, b, b_features, on_device=True):
    """Computes the similarity between two columns, by computing the inner product
    along two sparse feature matrices . Both a_features and b_features are
    required to be in canonical CSR format.

    Parameters
    -----------
    a : array of int
        Array of rowids to use in looking up a_features
    a_features: CSR matrix
        Sparse feature matrix
    b : array of int
        Array of rowids to use in looking up in b_features
    b_features: CSR matrix
        Sparse feature matrix
    on_device: bool
        Whether to compute on the GPU or CPU. Computing on the GPU will be
        faster, but requires that the a_features/b_features sparse matrices
        fit into GPU memory.
    """
    # run a JIT compiled version of this either on gpu/cpu with numba.
    # note that numba doesn't handle sparse matrix types, so we're splitting
    # out to the relevant cupy/numpy arrays for indptr/indices/data
    if on_device:
        threadsperblock = 32
        blockspergrid = (a.size + (threadsperblock - 1)) // threadsperblock
        output = cupy.zeros(len(a), dtype=a_features.data.dtype)
        _row_wise_inner_product_gpu[blockspergrid, threadsperblock](
            a,
            a_features.indptr,
            a_features.indices,
            a_features.data,
            b,
            b_features.indptr,
            b_features.indices,
            b_features.data,
            output,
        )
    else:
        output = numpy.zeros(len(a), dtype=a_features.data.dtype)
        _row_wise_inner_product_cpu(
            a,
            a_features.indptr,
            a_features.indices,
            a_features.data,
            b,
            b_features.indptr,
            b_features.indices,
            b_features.data,
            output,
        )

    return output


@numba.njit(parallel=True)
def _row_wise_inner_product_cpu(
    a, a_indptr, a_indices, a_data, b, b_indptr, b_indices, b_data, output
):
    for i in numba.prange(len(a)):
        output[i] = _inner_product_cpu(
            a[i], a_indptr, a_indices, a_data, b[i], b_indptr, b_indices, b_data
        )


@numba.cuda.jit
def _row_wise_inner_product_gpu(
    a, a_indptr, a_indices, a_data, b, b_indptr, b_indices, b_data, output
):
    i = numba.cuda.grid(1)
    if i < a.size:
        output[i] = _inner_product_gpu(
            a[i], a_indptr, a_indices, a_data, b[i], b_indptr, b_indices, b_data
        )


def _inner_product(a, a_indptr, a_indices, a_data, b, b_indptr, b_indices, b_data):
    # adapted from scipy:
    # https://github.com/scipy/scipy/blob/312b706c1d98980ed140adae943d41f9f7dc08f5/scipy/sparse/sparsetools/csr.h#L780-L854
    a_pos, a_end = a_indptr[a], a_indptr[a + 1]
    b_pos, b_end = b_indptr[b], b_indptr[b + 1]
    similarity = 0.0

    while a_pos < a_end and b_pos < b_end:
        a_j = a_indices[a_pos]
        b_j = b_indices[b_pos]
        if a_j == b_j:
            similarity += a_data[a_pos] * b_data[b_pos]
            a_pos += 1
            b_pos += 1
        elif a_j < b_j:
            a_pos += 1
        else:
            b_pos += 1

    return similarity


# JIT the _inner_product function to run on both CPU/GPU using numba
_inner_product_cpu = numba.njit(inline="always")(_inner_product)
_inner_product_gpu = numba.cuda.jit(device=True, inline=True)(_inner_product)


def _convert_features(features, metric, on_device):
    if on_device:
        # take a shallow copy to avoid mutating the input, but keep gpu
        # memory as low as possible. (also convert to coo_matrix if passed
        # a CSR etc)
        features = cupy.sparse.coo_matrix(features)
    else:
        if not isinstance(features, scipy.sparse.coo_matrix):
            # convert to host first if the sparse matrix is on the device
            if features.__class__.__module__.startswith("cupy"):
                features = features.get()
            # make sure we're a coo matrix
            if not isinstance(features, scipy.sparse.coo_matrix):
                features = scipy.sparse.coo_matrix(features)

    # Normalizes the matrix so that we can compute the distance metric
    # with only the inner product
    np = cupy if on_device else numpy
    if metric == "tfidf":
        features = _normalize(_tfidf_weight(features, np), np)
    elif metric == "cosine":
        features = _normalize(features, np)
    elif metric != "inner":
        raise ValueError(f"unknown distance metric {metric}")

    # we need features in CSR format to do the row lookup
    return features.tocsr()


def _tfidf_weight(X, np):
    N = float(X.shape[0])
    idf = np.log(N / np.bincount(X.col))
    X.data = X.data * idf[X.col]
    return X


def _normalize(X, np):
    X.data = X.data / np.sqrt(np.bincount(X.row, X.data ** 2))[X.row]
    return X
