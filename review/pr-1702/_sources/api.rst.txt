*****************
API Documentation
*****************

Workflow Constructors
---------------------

.. currentmodule:: nvtabular.workflow.workflow

.. autosummary::
   :toctree: generated

   Workflow
   WorkflowNode

.. currentmodule:: nvtabular.ops


Categorical Operators
---------------------

.. autosummary::
   :toctree: generated

   Bucketize
   Categorify
   DropLowCardinality
   HashBucket
   HashedCross
   TargetEncoding


Continuous Operators
--------------------

.. autosummary::
   :toctree: generated

   Clip
   LogOp
   Normalize
   NormalizeMinMax


Missing Value Operators
-----------------------

.. autosummary::
   :toctree: generated

   Dropna
   FillMissing
   FillMedian


Row Manipulation Operators
--------------------------

.. autosummary::
   :toctree: generated

   DifferenceLag
   Filter
   Groupby
   JoinExternal
   JoinGroupby


Schema Operators
----------------

.. autosummary::
   :toctree: generated

   AddMetadata
   AddProperties
   AddTags
   Rename
   ReduceDtypeSize
   TagAsItemFeatures
   TagAsItemID
   TagAsUserFeatures
   TagAsUserID


List Operators
--------------

.. autosummary::
   :toctree: generated

   ListSlice
   ValueCount


Vector Operators
----------------

.. autosummary::
   :toctree: generated

   ColumnSimilarity


User-Defined Function Operators
-------------------------------

.. autosummary::
   :toctree: generated

   LambdaOp


Operator Base Classes
---------------------

.. autosummary::
   :toctree: generated

   Operator
   StatOperator