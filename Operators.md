Operator Support
=====================

NVTabular provides operators for preprocessing and feature engineering.  While the initial set is limited, we are currently working to expand upon the available operator set.

Currently support is available for:

| Operator Name | Data Type | Major Options | Description |
| -----------   | --------- | ------------- | ----------- |
| GroupBy       | [Categorical],[Continuous]| min, max, mean, stdev, distance from min, max & mean | For a set of categorical columns, perform a groupby operation on that set and calculate the min, max, mean, stdev, and distance from min, max & mean for one or more continuous variables, including the target variable.  Binary & Ordinal continuous variables are allowed as well.  Lambda as an option here makes sense as well but is slightly more complex.|
| Join          | [Categorical] | left, right, inner, outer | Join a table or file by categorical fields |
| Encoding_Numerical | Categorical | Full, Frequency Filtered | Encodes categories into a contiguous space from 0 to cardinality, reserving 0 for missing/null values.|
| Normalize | Continuous | Standard, MinMax, Mean | Normalizes the data based on the chosen normalization type. |
| Gaussify | Continuous | GaussRank | Transforms the distribution of a continuous variable into a Gaussian |
| Log | Continuous | | Transforms an exponential distribution into one that is linear |
| Fill_Missing | Continuous / Categorical | Fixed Value, Mean, Median (of medians) / Fixed Value, Mode, Unique | Replace missing values in a column with another appropriate value |

