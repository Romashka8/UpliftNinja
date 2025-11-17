# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


class BasePipeline(BaseEstimator, TransformerMixin):
    def __init__(self,
                 steps
                 ):
        self.steps = steps
        self.transformers = [step[1] for step in steps[:-1]]
        self.model = steps[-1][1]

    def fit(self, data, target=None, treatment=None):
        transformed_data = data.copy()
        for transformer in self.transformers:
            transformed_data = transformer.fit_transform(
                transformed_data, target)
        if hasattr(self.model, 'fit'):
            self.model.fit(transformed_data, target, treatment)
        else:
            self.model.fit(transformed_data, target, treatment)
        return self

    def transform(self, data):
        transformed_data = data.copy()
        for transformer in self.transformers:
            transformed_data = transformer.transform(transformed_data)
        if not hasattr(self.model, 'predict'):
            transformed_data = self.model.transform(transformed_data)
        return transformed_data

    def fit_transform(self, data, target=None, treatment=None):
        self.fit(data, target, treatment)
        return self.transform(data)

    def predict(self, data):
        transformed_data = self.transform(data)
        if hasattr(self.model, 'predict'):
            transformed_data = transformed_data.values.copy()
            return self.model.predict(transformed_data)
        raise AttributeError(
            f"Last step does'n support 'predict' method! Last step: {print(self.model)}")

    def score(self, data, target):
        transformed_data = self.transform(data)
        if hasattr(self.model, 'score'):
            transformed_data = transformed_data.values.copy()
            return self.model.score(transformed_data, y)
        raise AttributeError(
            f"Last step does'n support 'score' method! Last step: {print(self.model)}")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
