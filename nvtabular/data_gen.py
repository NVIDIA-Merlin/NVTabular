import cupy
import cudf
import numpy as np
from scipy import stats
from scipy.stats import powerlaw, uniform


class UniformDistro:
    def create_col(self, num_rows, dtype=np.float32, min_val=0, max_val=1):
        ser = cudf.Series(cupy.random.uniform(min_val, max_val, size=num_rows))
        return ser

    def verify(self, pandas_series):
        return stats.kstest(pandas_series, uniform().cdf)


class PowerLawDistro:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def create_col(self, num_rows, dtype=np.float32, min_val=0, max_val=1):
        gamma = 1 - self.alpha
        ser = cudf.Series(cupy.random.uniform(0.0, 1.0, size=num_rows))
        factor = (cupy.power(max_val, gamma) - cupy.power(min_val, gamma)) + cupy.power(
            min_val, gamma
        )
        ser = ser * factor.item()
        exp = 1.0 / gamma
        ser = ser.pow(exp)
        # add in nulls if requested
        # select indexes
        return ser.astype(dtype)

    def verify(self, pandas_series):
        return stats.kstest(pandas_series, powerlaw(self.alpha).cdf)


class DatasetGen:
    def __init__(self, distribution):
        assert distribution is not None
        self.dist = distribution

    # need to be able to generate from scratch
    def create_conts(self, size, num_cols=1):
        df = cudf.DataFrame()
        for x in range(num_cols):
            ser = self.dist.create_col(size)
            ser.name = f"CONT_{x}"
            df = cudf.concat([df, ser], axis=1)
        return df

    def create_cats(self, size, num_cols=1, cardinality=[1]):
        """
        size = number of rows
        num_cols = how many columns you want produced
        cardinality = a list of values representing the desired cardinalities for columns
        dist = distribution type to use for creating columns ("powerlaw", "uniform")
        """
        # should alpha also be exposed? related to dist... should be part of that
        assert len(cardinality) == num_cols
        df = cudf.DataFrame()
        for x in range(num_cols):
            ser = self.dist.create_col(
                size, dtype=np.long, min_val=1.0, max_val=cardinality[x]
            ).ceil()
            ser.name = f"CAT_{x}"
            df = cudf.concat([df, ser], axis=1)
        return df

    def create_df(self, size, num_conts, num_cats, cat_cardinality=[], dist=PowerLawDistro()):
        df = cudf.DataFrame()
        df = cudf.concat([df, self.create_conts(size, num_cols=num_conts)], axis=1)
        df = cudf.concat(
            [df, self.create_cats(size, num_cols=num_cats, cardinality=cat_cardinality)], axis=1
        )
        return df

    def verify_df(self, df_to_verify):
        sts = []
        ps = []
        for col in df_to_verify.columns:
            st_df, p_df = self.dist.verify(df_to_verify[col].to_pandas())
            sts.append(st_df)
            ps.append(p_df)
        return sts, ps
