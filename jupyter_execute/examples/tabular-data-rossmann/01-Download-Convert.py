#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2021 NVIDIA Corporation. All Rights Reserved.
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
# ==============================================================================


# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Preprocessing the Rossmann Store Sales Dataset
# Here we implement some feature engineering outlined by FastAI in [their example solution](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson6-rossmann.ipynb) to the [Kaggle Rossmann Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales) using Rapids cuDF library (GPU DataFrame library). We've simplified some sections and left out some of the documentation to keep things neat, so feel free to consult the original preprocessing [notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/rossman_data_clean.ipynb) for explanations of the feature engineering going on.

# In[2]:


import os
import pandas as pd

# Get dataframe library - cudf or pandas
from merlin.core.dispatch import get_lib
df_lib = get_lib()

from merlin.core.utils import download_file


# In[3]:


INPUT_DATA_DIR = os.environ.get(
    "INPUT_DATA_DIR", os.path.expanduser("~/nvt-examples/rossmann/")
)
OUTPUT_DATA_DIR = os.environ.get(
    "OUTPUT_DATA_DIR", os.path.expanduser("~/nvt-examples/data/")
)


# The cell below will download the data into the `INPUT_DATA_DIR` if you do not have it already.

# In[4]:


if not os.path.isfile(os.path.join(INPUT_DATA_DIR, "rossmann.tgz")):
    download_file(
        "https://files.fast.ai/part2/lesson14/rossmann.tgz",
        os.path.join(INPUT_DATA_DIR, "rossmann.tgz"),
    )


# Let's read each csv file as a cudf dataframe.

# In[6]:


def read_table(table_name):
    return df_lib.read_csv(os.path.join(INPUT_DATA_DIR, f"{table_name}.csv"))


train = read_table("train")
store = read_table("store")
store_states = read_table("store_states")
state_names = read_table("state_names")
googletrend = read_table("googletrend")
weather = read_table("weather")
test = read_table("test")


# StateHoliday column indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. Below, we turn state Holidays to booleans to make them more convenient for modeling.

# In[7]:


train.StateHoliday = train.StateHoliday != "0"
test.StateHoliday = test.StateHoliday != "0"


# Here, new features are added by extracting dates and state names from the given data. Also all instances of state name 'NI' are replaced to match the usage in the rest of the data: 'HB,NI'. 
# 
# We create a new data frame `trend_de` from the Google trends data that has a special category for the whole of the Germany.

# In[8]:


googletrend["Date"] = googletrend.week.str.split(" - ", expand=True)[0]
googletrend["State"] = googletrend.file.str.split("_", expand=True)[2]
googletrend["State"] = googletrend.State.where(googletrend["State"] != "NI", "HB,NI")
trend_de = googletrend.loc[googletrend.file == "Rossmann_DE"].copy()


# The following extracts particular date fields from a complete datetime for the purpose of constructing categoricals. When working with date-time features, you should always consider this feature extraction step. You can capture the trend/cyclical behavior as a function of time at any of these granularities by expanding your date-time into these additional fields. Here, we create `Day, Week, Month and Year` features, and add these to every table with a date field. Note that cudf currently does not have `dt.week` property, thus, we applied a custom method to create `Week` column.

# In[9]:


for df in (weather, googletrend, train, test, trend_de):
    df["Date"] = df["Date"].astype("datetime64[ns]")
    df["Month"] = df.Date.dt.month
    df["Day"] = df.Date.dt.day
    df["Year"] = df.Date.dt.year.astype(str) + "-01-01"
    df["Week"] = (((df["Date"] - df["Year"].astype("datetime64[ns]")).dt.days) / 7).astype(
        "int16"
    ) + 1
    df["Week"] = df.Week.where(df["Week"] != 53, 52)
    df["Year"] = df.Date.dt.year


# We will use the function below to perform left outer join operation. The suffixes argument describes the naming convention for duplicate fields.

# In[10]:


def merge(df, right, left_on, right_on=None, suffix=None):
    df = df.merge(
        right,
        how="left",
        left_on=left_on,
        right_on=right_on or left_on,
        suffixes=("", suffix or "_y"),
    )
    return df


# This function will be used to drop redundant or unwanted columns generated via join operations.

# In[11]:


def drop_cols(gdf):
    for c in gdf.columns:
        if c.endswith("_y"):
            if c in gdf.columns:
                gdf.drop(c, inplace=True, axis=1)
    return gdf


# Now we can outer join all of our data into a single dataframe that we will use to train our DL model.

# In[12]:


weather = merge(weather, state_names, "file", right_on="StateName")
store = merge(store, store_states, "Store")
train_df = merge(train, store, "Store")
test_df = merge(test, store, "Store")


# In[13]:


train_df = merge(train_df, googletrend, ["State", "Year", "Week"])
test_df = merge(test_df, googletrend, ["State", "Year", "Week"])
train = test = googletrend = None


# In[14]:


train_df = merge(train_df, trend_de, ["Year", "Week"], right_on=["Year", "Week"], suffix="_DE")
test_df = merge(test_df, trend_de, ["Year", "Week"], right_on=["Year", "Week"], suffix="_DE")


# In[15]:


train_df = drop_cols(train_df)
train_df = merge(train_df, weather, ["State", "Date"], right_on=["State", "Date"], suffix="_y")
test_df = drop_cols(test_df)
test_df = merge(test_df, weather, ["State", "Date"], right_on=["State", "Date"])


# In[16]:


# drop the redundant columns with `_y` suffix.
train_df = drop_cols(train_df)
test_df = drop_cols(test_df)


# Next we'll fill in missing values to avoid complications with NA's. Many models have problems when missing values are present, so it's important to think about how to deal with them. Here, we are picking arbitrary signal values and filling the missing values with them.

# In[17]:


for df in [train_df, test_df]:
    df["CompetitionOpenSinceYear"] = df.CompetitionOpenSinceYear.fillna(1900).astype("int32")
    df["CompetitionOpenSinceMonth"] = df.CompetitionOpenSinceMonth.fillna(1).astype("int32")
    df["Promo2SinceYear"] = df.Promo2SinceYear.fillna(1900).astype("int32")
    df["Promo2SinceWeek"] = df.Promo2SinceWeek.fillna(1).astype("int32")


# Next we'll extract features "CompetitionOpenSince" and "CompetitionDaysOpen". 

# In[18]:


for df in [train_df, test_df]:
    df["year"] = df["CompetitionOpenSinceYear"]
    df["month"] = df["CompetitionOpenSinceMonth"]
    df["day"] = "15"
    df["CompetitionOpenSince"] = df_lib.to_datetime(df[["year", "month", "day"]])
    df["CompetitionDaysOpen"] = (df["Date"] - df["CompetitionOpenSince"]).dt.days


# Let's replace some erroneous / outlying data.

# In[19]:


for df in [train_df, test_df]:
    df.loc[df.CompetitionDaysOpen < 0, "CompetitionDaysOpen"] = 0
    df.loc[df.CompetitionOpenSinceYear < 1990, "CompetitionDaysOpen"] = 0


# We'll now add "CompetitionMonthsOpen" field, limiting the maximum to two years to limit number of unique categories.

# In[20]:


for df in [train_df, test_df]:
    df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"] // 30
    df.loc[df.CompetitionMonthsOpen > 24, "CompetitionMonthsOpen"] = 24


# Same process is applied for Promo dates. We'll create "Promo2Days"  and "Promo2Weeks" fields, limiting the maximum to 25 weeks to limit number of unique categories.

# In[21]:


for df in [train_df, test_df]:
    df["Promo2SinceYear_tmp"] = df.Promo2SinceYear.astype(str) + "-01-01"
    dt = df_lib.to_datetime(df.Promo2SinceYear_tmp, format="%Y").astype("int64") // 10 ** 9
    dt += 7 * 24 * 3600 * df.Promo2SinceWeek
    df["Promo2Since"] = df_lib.to_datetime(dt * 10 ** 9)
    df["Promo2Days"] = (df["Date"] - df["Promo2Since"]).dt.days


# In[22]:


for df in [train_df, test_df]:
    df.loc[df.Promo2Days < 0, "Promo2Days"] = 0
    df.loc[df.Promo2SinceYear < 1990, "Promo2Days"] = 0
    df["Promo2Weeks"] = df["Promo2Days"] // 7
    df.loc[df.Promo2Weeks < 0, "Promo2Weeks"] = 0
    df.loc[df.Promo2Weeks > 25, "Promo2Weeks"] = 25


# In[23]:


df = train_df.append(test_df, ignore_index=True)


# In[24]:


# let's drop these dummy columns.
df.drop(["year", "month", "day", "Promo2SinceYear_tmp"], inplace=True, axis=1)
train_df.drop(["year", "month", "day", "Promo2SinceYear_tmp"], inplace=True, axis=1)
test_df.drop(["year", "month", "day", "Promo2SinceYear_tmp"], inplace=True, axis=1)


# In[25]:


# # cast SchoolHoliday to int32.
df["SchoolHoliday"] = df["SchoolHoliday"].astype("int32")


# This is modififed version of the original `get_elapsed` function from [rossmann_data_clean nb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/rossman_data_clean.ipynb). The `get_elapsed` function is defined for cumulative counting across a sorted dataframe. Given a particular field to monitor, this function starts tracking time since the last occurrence of that field. When the field is seen again, the counter is set to zero.
# 
# This section is still not working as intended with cuDF, therefore we are converting the cudf df to pandas df to perform the `bfill` and `ffill` operations below. We can explain the code below by looking at School Holiday as an exp. <i>SchoolHoliday</i> column indicates if the (Store, Date) was affected by the closure of public schools. First every row of the dataframe is sorted in order of store and date. Then, we will add to the dataframe the days since seeing a School Holiday, and the days until another holiday.

# In[26]:


# ops: masking, ffill, bfill, timedelta
df = df.sort_values(by=["Store", "Date"])
# We convert cudf dataframe to the pandas df be able to do `bfill` and `ffill` operations below.
if df_lib == pd:
    pdf = df
else:
    pdf = df.to_pandas()


# In[27]:


# first build a mask indicating where stores start and end
first_indices = pdf.Store.diff() != 0
last_indices = pdf.Store.diff().iloc[1:].append(pd.Series([1]))
last_indices.index = first_indices.index
idx_mask = ~(first_indices | last_indices)

event_fields = ["SchoolHoliday", "StateHoliday", "Promo"]
for field in event_fields:
    # use the mask from above to mask save dates from the start and end
    # of a given store's range, as well as all dates that have an event
    pdf["tmp"] = pdf.Date
    pdf.loc[(pdf[field] == 0) & idx_mask, "tmp"] = float("nan")
    # then use ffill and bbfill to give the input to the time delta
    pdf["After" + field] = pdf.tmp.ffill()
    pdf["Before" + field] = pdf.tmp.bfill()

    # compute deltas between bfilled and ffilled dates and the current date
    pdf["After" + field] = (pdf["Date"] - pdf["After" + field]).astype("timedelta64[D]")
    pdf["Before" + field] = (pdf["Before" + field] - pdf["Date"]).astype("timedelta64[D]")

# get rid of our dummy column
pdf = pdf.drop(columns=["tmp"])

# let's convert pandas back to cudf df
if df_lib == pd:
    df = pdf
else:
    df = df_lib.from_pandas(pdf)


# In[28]:


pd.DataFrame(pdf.dtypes).to_csv("dtypes_none.csv")


# In[29]:


# Set the active index to Date.
df = df.set_index("Date")


# Next we'll apply window calculations in cudf to calculate rolling quantities. Here, we're sorting by date (sort_index()), grouping by Store column, and counting the number of events of interest (sum()) defined in columns in the following week (rolling()). We do the same in the opposite direction.

# In[30]:


event_fields = ["SchoolHoliday", "StateHoliday", "Promo"]
bwd = df[["Store"] + event_fields].sort_index().groupby("Store").rolling(7, min_periods=1).sum()
fwd = (
    df[["Store"] + event_fields]
    .sort_index(ascending=False)
    .groupby("Store")
    .rolling(7, min_periods=1)
    .sum()
)


# Next we drop the Store indices grouped together in the window function.

# In[31]:


for d in (bwd, fwd):
    d.drop("Store", 2, inplace=True)
    d.reset_index(inplace=True)


# In[32]:


df.reset_index(inplace=True)


# 
# Let's merge these values onto the df, and drop the unwanted columns.

# In[33]:


df = df.merge(
    bwd, left_on=["Date", "Store"], right_on=["Date", "Store"], how="left", suffixes=["", "_bw"]
)
df = df.merge(
    fwd, left_on=["Date", "Store"], right_on=["Date", "Store"], how="left", suffixes=["", "_fw"]
)


# In[34]:


train_df = merge(train_df, df, ["Store", "Date"], right_on=["Store", "Date"])
test_df = merge(test_df, df, ["Store", "Date"], right_on=["Store", "Date"])


# In[35]:


train_df = drop_cols(train_df)
test_df = drop_cols(test_df)


# Next, we are removing all instances where the store had zero sale / was closed, and we sort our training set by Date.

# In[36]:


train_df = train_df[train_df.Sales != 0]
train_df = train_df.sort_values(by="Date", ascending=True)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# In[37]:


# determine a cut-off point to create a validation dataset.
cut = train_df["Date"][(train_df["Date"] == train_df["Date"][len(test_df)])].index.max()
cut


# In[38]:


# split the train_df as training and validation data sets.
num_valid = cut
valid_df = train_df[-num_valid:]
train_df = train_df[:-num_valid]


# Let's save our preprocessed files as csv files to be used in [Rossmann examples](https://github.com/NVIDIA/NVTabular/blob/main/examples/rossmann/) for further preprocessing with NVTabular, then for model training and testing with PyT, TF and FastAI frameworks.

# In[39]:


get_ipython().system('mkdir -p $OUTPUT_DATA_DIR')

train_df.to_csv(os.path.join(OUTPUT_DATA_DIR, "train.csv"), index=False)
valid_df.to_csv(os.path.join(OUTPUT_DATA_DIR, "valid.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DATA_DIR, "test.csv"), index=False)


# In[40]:


get_ipython().system('ls $OUTPUT_DATA_DIR')

