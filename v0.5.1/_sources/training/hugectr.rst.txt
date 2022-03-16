Accelerated Training with HugeCTR
=================================

A real-world production model serves hundreds of millions of users,
which contains embedding tables that can exceed the memory of a single
GPU with up to 100GB to 1TB in size. Large embedding tables in deep
learning recommender system models can be challenging.

To combat that challenge, we’ve developed HugeCTR, which is an open-source deep learning framework that is a highly optimized library
written in CUDA C++, specifically for recommender systems. It supports
an optimized dataloader and is able to scale embedding tables using
multiple GPUs and nodes. As a result, there’s no embedding table size
limitation. HugeCTR also offers the following:

-  Model oversubscription for training embedding tables with
   single nodes that don’t fit within the GPU or CPU memory (only
   required embeddings are prefetched from a parameter server per
   batch).
-  Asynchronous and multithreaded data pipelines.
-  A highly optimized data loader.
-  Implementation of common architectures such as Wide&Deep and DLRM.
-  Support for data formats such as parquet and binary.
-  Easy configuration using JSON or the Python API.

When training is accelerated with HugeCTR, the following happens:

1. The required libraries are imported in which the HugeCTR lib
   directory is specified as follows:

   .. code:: python

      import sys
      sys.path.append("/usr/local/hugectr/lib")
      from hugectr import Session, solver_parser_helper, get_learning_rate_scheduler

2. The JSON configuration file is specified, which defines the model
   architecture.

   .. code:: python

      # Set config file
      json_file = "dlrm_fp32_64k.json"

   The JSON file defines the input layers as follows:

   -  ``slot_size_array`` is the cardinality of categorical input
      features
   -  ``source`` is a text file that contains filenames for training
   -  ``eval_source`` is a text file that contains filenames for
      evaluation
   -  ``label``-``label_dim`` provides the number of target columns
   -  ``dense``-``label_dim`` provides the number of continuous input
      features
   -  ``sparse``-``label_dim`` provides the number of categorical input
      features

.. code:: python

   # Part of JSON config
   "layers": [
      {
      "name": "data",
      "type": "Data",
      "format": "Parquet",
      "slot_size_array": [10000000, 10000000, 3014529, 400781, 11, 2209, 11869, 148, 4, 977, 15, 38713, 10000000, 10000000, 10000000, 584616, 12883, 109, 37, 17177, 7425,             20266, 4, 7085, 1535, 64],
      "source": "/raid/criteo/tests/test_dask/output/train/_file_list.txt",
      "eval_source": "/raid/criteo/tests/test_dask/output/valid/_file_list.txt",
      "check": "None",
      "label": {
          "top": "label",
          "label_dim": 1
      },
      "dense": {
          "top": "dense",
          "dense_dim": 13
      },
      "sparse": [
          {
          "top": "data1",
          "type": "LocalizedSlot",
          "max_feature_num_per_sample": 30,
          "max_nnz": 1,
          "slot_num": 26
          }
      ]
   },

3. The solver configuration is defined. The batch_sizes for training,
   validation, and GPUs are specified in the solver
   configuration.

   .. code:: python

      # Set solver config
      solver_config = solver_parser_helper(seed = 0,
                                           batchsize = 16384,
                                           batchsize_eval = 16384,
                                           vvgpu = [[0,1,2,3,4,5,6,7]],
                                           repeat_dataset = True

      )

4. The learning rate schedule in the JSON file and HugeCTR session is
   initialized.

   .. code:: python

      # Set learning rate
      lr_sch = get_learning_rate_scheduler(json_file)
      # Train model
      sess = Session(solver_config, json_file)
      sess.start_data_reading()

5. The dataset is iterated for 5000 steps and the model is trained.

   .. code:: python

      for i in range(5000):
         lr = lr_sch.get_next()
         sess.set_learning_rate(lr)
         sess.train()
         if (i%100 == 0):
           loss = sess.get_current_loss()
           print("[HUGECTR][INFO] iter: {}; loss: {}".format(i, loss))
         if (i%3000 == 0 and i != 0):
           metrics = sess.evaluation()
           print("[HUGECTR][INFO] iter: {}, {}".format(i, metrics))

Additional examples can be found `here`_.

.. _here: https://github.com/NVIDIA/NVTabular/tree/main/examples/hugectr
