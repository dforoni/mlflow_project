# Custom Transformer Class

'''In order to demonstrate the use of a custom transformer we will create a new feature
    based upon the ratio of the resting blood pressure to the maximum blood pressure.
    This feature will be created as a new class and saved into a separate file 
    so it can be output to the MLflow tracking server to be used during deployment
    in combination with the saved model.'''
 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NewFeatureTransformer(BaseEstimator, TransformerMixin):
        def fit(self, x, y=None):
            return self
        def transform(self, x):
            x['ratio'] = x['thalach']/x['trestbps']
            x = pd.DataFrame(x.loc[:, 'ratio'])
            return x.values