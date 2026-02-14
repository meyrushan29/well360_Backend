"""
Preprocessing Pipeline for Hydration Models
============================================

This module handles data preprocessing for both:
1. Tabular data (form predictions)
2. Image data (lip analysis)
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from typing import List

from core.utils import setup_logging

LOG = setup_logging()


# ======================================================
# TABULAR DATA PREPROCESSING
# ======================================================

class PreprocessorWithFeatureNames:
    """
    Wrapper for ColumnTransformer that stores feature names.
    Defined at module level to be picklable.
    """
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self._feature_names = None
    
    def fit(self, X, y=None):
        self.preprocessor.fit(X, y)
        self._compute_feature_names(X)
        return self
    
    def transform(self, X):
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def _compute_feature_names(self, X):
        """Compute feature names after transformation"""
        feature_names = []
        
        # Numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        feature_names.extend(numeric_features)
        
        # Categorical features (after one-hot encoding)
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_features:
            try:
                cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
                feature_names.extend(cat_feature_names)
            except:
                # Fallback
                feature_names.extend([f"cat_{i}" for i in range(len(categorical_features))])
        
        self._feature_names = feature_names
    
    def get_feature_names(self):
        """Return feature names after transformation"""
        return self._feature_names if self._feature_names else []


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a preprocessing pipeline for tabular hydration data.
    
    Args:
        X: Training dataframe
        
    Returns:
        Fitted ColumnTransformer
    """
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Wrap preprocessor with feature name tracking
    wrapped_preprocessor = PreprocessorWithFeatureNames(preprocessor)
    
    LOG.info(f"Preprocessor created with {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
    
    return wrapped_preprocessor
