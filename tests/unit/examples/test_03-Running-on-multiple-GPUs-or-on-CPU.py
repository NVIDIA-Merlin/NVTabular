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
import nest_asyncio
import pytest
from testbook import testbook

from tests.conftest import REPO_ROOT

from merlin.core.compat import dask_cudf

if not dask_cudf:
    pytest.mark.skip(reason="could not import dask_cudf successfully")

nest_asyncio.apply()


def test_example_03():
    with testbook(
        REPO_ROOT / "examples" / "03-Running-on-multiple-GPUs-or-on-CPU.ipynb",
        execute=False,
        timeout=180,
    ) as tb:
        tb.inject(
            """
            import os
            from unittest.mock import patch
            from merlin.datasets.synthetic import generate_data
            mock_train, mock_valid = generate_data(
                input="movielens-1m",
                num_rows=1000,
                set_sizes=(0.8, 0.2)
            )
            input_path = os.environ.get(
                "INPUT_DATA_DIR",
                os.path.expanduser("~/merlin-framework/movielens/")
            )
            from pathlib import Path
            Path(f'{input_path}ml-1m').mkdir(parents=True, exist_ok=True)
            mock_train.compute().to_parquet(f'{input_path}ml-1m/train.parquet')
            mock_train.compute().to_parquet(f'{input_path}ml-1m/valid.parquet')

            p1 = patch(
                "merlin.datasets.entertainment.get_movielens",
                return_value=[mock_train, mock_valid]
            )
            p1.start()

            """
        )
        for cell in tb.cells:
            cell.source.replace(
                "device_memory_limit=device_limit", "# device_memory_limit=device_limit"
            )
            cell.source.replace("rmm_pool_size=", "# rmm_pool_size=")
        tb.execute()
