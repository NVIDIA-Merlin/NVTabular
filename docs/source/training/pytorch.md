Accelerated Training with PyTorch
=================================

When training pipelines with PyTorch, the dataloader cannot prepare sequential batches fast enough, so the GPU is not fully utilized. To combat this issue, we’ve developed a highly customized tabular dataloader, `TorchAsyncItr`, to accelerate existing pipelines in PyTorch. The NVTabular dataloader is capable of:

* removing bottlenecks from dataloading by processing large chunks of data at a time instead of item by item
* processing datasets that don’t fit within the GPU or CPU memory by streaming from the disk
* reading data directly into the GPU memory and removing CPU-GPU communication
* preparing batch asynchronously into the GPU to avoid CPU-GPU communication
* supporting commonly used formats such as parquet
* integrating easily into existing PyTorch training pipelines by using a similar API as the native PyTorch dataloader

When `TorchAsyncItr` accelerates training with PyTorch, the following happens:

1. The required libraries are imported.
   
   ```python
   import torch
   from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader
   ```

2. The `TorchAsyncItr` iterator is initialized.
   The input is a NVTabular dataset that uses a list of file names. The NVTabular dataset is an abstraction layer that iterates over the full dataset in chunks. The dataset        schema is defined in which `cats` are the column names for the categorical input features, `conts` are the column names for the continuous input features, and `labels` are      the column names for the target. Each  parameter should be formatted as a list of strings. The batch size is also specified.
   
   ```python
   TRAIN_PATHS = glob.glob(‘./train/*.parquet’)
   train_dataset = TorchAsyncItr(
      nvt.Dataset(TRAIN_PATHS), 
      cats=CATEGORICAL_COLUMNS, 
      conts=CONTINUOUS_COLUMNS, 
      labels=LABEL_COLUMNS,
      batch_size=BATCH_SIZE
   )
   ```
   
3. `TorchAsyncItr` is wrapped as `DLDataLoader`.

   ```python
   train_loader = DLDataLoader(
      train_dataset, 
      batch_size=None, 
      collate_fn=collate_fn, 
      pin_memory=False, 
      num_workers=0
   )
   ```

4. If a `torch.nn.Module` model was created, `train_loader` can be used in the same way as the PyTorch dataloader. 

   ```python
   ...
   model = get_model()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
   for x_cat, x_cont, y in iter(dataloader):
      y_pred = model(x_cat, x_cont)
      loss = loss_func(y_pred, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
   ```

5. The `TorchAsyncItr` dataloader can be initialized for the validation dataset using the same structure.  

You can find additional examples in our repository such as [MovieLens](../examples/getting-started-movielens/) and [Criteo](
../examples/scaling-criteo/).
