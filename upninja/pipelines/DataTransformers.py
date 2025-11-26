# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd

from catboost import Pool

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

class CatBoostTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 **catboost_params
                 ):
        self.catboost_params = catboost_params

    def fit(self, data, target=None):
        return self

    def transform(self, data, target=None):
        data_transformed = Pool(data=data, label=target)
        return data_transformed

    def fit_transform(self, data, target=None):
        return self.fit(data, target).transform(data, target)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


class HillstromTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 cat_columns=["history_segment", "zip_code", "channel"]
                 ):
        self.cat_columns = cat_columns
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, data, target=None):
        self.encoder.fit(data[self.cat_columns])
        return self

    def transform(self, data):
        data_transformed = self.encoder.transform(data[self.cat_columns])
        data_transformed = pd.DataFrame(
            data_transformed,
            index=data.index,
            columns=self.encoder.get_feature_names_out(
                self.cat_columns))
        return pd.concat(
            [data_transformed, data.drop(self.cat_columns, axis=1)], axis=1)

    def fit_transform(self, data, target=None):
        return self.fit(data, target).transform(data)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


class AddControl(BaseEstimator, TransformerMixin):
    def __init__(self,
                 control,
                 control_name="control"
                 ):
        self.control = control
        self.control_name = control_name

    def fit(self, data, target=None):
        return self

    def transform(self, data):
        data[f"{control_name}"] = control
        return data

    def fit_transform(self, data, target=None):
        return self.fit(data, target).transform(data)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
