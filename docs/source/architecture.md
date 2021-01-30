Architecture
============

![NVTabular Workflow](../images/nvt_workflow.png)

The NVTabular engine uses the [RAPIDS](http://www.rapids.ai) [Dask-cuDF library](https://github.com/rapidsai/dask-cuda), which provides the bulk of the functionality for accelerating dataframe operations on the GPU and scaling across multiple GPUs. NVTabular provides functionality commonly found in deep learning recommendation workflows, allowing you to focus on what you want to do with your data, and not how you need to do it. NVTabular also provides a template for our core compute mechanism, which is referred to as Operations (ops), allowing you to build your own custom ops from cuDF and other libraries.

Once NVTabular is installed, you can set up a workflow as follows:

```python
import nvtabular as nvt
workflow = nvt.Workflow(
    cat_names=["user_id", "item_id", "city"],
    cont_names=["age", "time_of_day", "item_num_views"],
    label_name=["label"]
)
```

For additional information about installing NVTabular, see https://nvidia.github.io/NVTabular/main/Introduction.html#installation. With the workflow in place, we can now explore the library in detail.

## Operations

Operations are a reflection of the way in which compute happens on the GPU across large datasets. There are two types of compute:

* the type that touches the entire dataset (or some large chunk of it)
* the type that operates on a single row

Operations split the compute into two phases:

* Statistics Gathering is the first phase where operations that cross the row boundary can occur. An example of this would be in the Normalize op that relies on two statistics: mean and standard deviation. To normalize a row, we must first have these two values calculated using a Dask-cudf graph.
* Apply is the second phase that uses the statistics, which were created earlier, to modify the dataset and transform the data. NVTabular allows for the application of transforms and not only during the modification of the dataset, but also during dataloading. The same transforms can also be applied with Inference.

```python
# by default, the op will be applied to _all_
# columns of the associated variable type
workflow.add_cont_preprocess(nvt.ops.Normalize())

dataset = nvt.Dataset("/path/to/data.parquet")

# record stats, transform the dataset, and export
# the transformed data to a parquet file
proc.apply(dataset, shuffle=nvt.io.Shuffle.PER_WORKER, output_path="/path/to/export/dir")
```

Dask-cuDF does the scheduling to help optimize the task graph by providing an optimal solution to whatever GPUs you have configured.

## A Higher Level of Abstraction

The NVTabular code is targeted at the operator level and not the dataframe level, which provides a method for specifying the operation you want to perform, as well as the columns or type of data that you want to perform it on. There's an explicit distinction between feature engineering ops and preprocessing ops. Feature engineering ops create new variables and preprocessing ops transform data more directly to make it ready for the model to which it’s feeding. While the type of computation involved in these two stages is often similar, we want to allow for the creation of new features that will then be preprocessed in the same way as other input variables.

Two main data types are currently supported: categorical variables and continuous variables. Feature engineering ops explicitly take one or more continuous or categorical columns as input and produce one or more columns of a specific type. By default, the input columns used to create the new feature are also included in the output. However, this can be overridden with the [replace] keyword in the operator. This is extended to multi-hot categoricals, as well as high cardinality categoricals, which must be treated differently due to memory constraints.

Preprocessing operators take in a set of columns of the same type and perform the operation across each column, transforming the output during the final operation into a long tensor in the case of categorical variables or a float tensor in the case of continuous variables. Preprocessing operations replace the column values with their new representation by default, but this can be overriden.

```python
# same example as before, but now only apply normalization
# to `age` and `item_num_views` columns, which will create
# new columns `age_normalize` and `item_num_views_normalize`
workflow.add_cont_preprocess(nvt.ops.Normalize(columns=["age", "item_num_views"], replace=False))

dataset = nvt.Dataset("/path/to/data.parquet")
proc.apply(dataset, shuffle=nvt.io.Shuffle.PER_WORKER, output_path="/path/to/export/dir")
```

Operators may also be chained to allow for more complex feature engineering or preprocessing. The chaining of operators is done by creating a list of operators. By default, only the final operator in a chain that includes preprocessing will be included in the output with all other intermediate steps implicitly dropped.

```python
# Replace negative and missing values with 0 and then take log(1+x)
workflow.add_cont_feature([FillMissing(), Clip(min_value=0), LogOp()])

# then normalize
workflow.add_cont_preprocess(Normalize())
```
