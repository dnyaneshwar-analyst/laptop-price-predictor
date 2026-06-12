"""
Core feature engineering and pipeline building functions.
Shared between training script and inference (Streamlit) app.
"""

from sklearn.preprocessing import (
    RobustScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


def convert_memory(x):
    """Convert memory strings like '256GB', '1TB' to GB as float."""
    if pd.isna(x):
        return 0.0
    x = str(x).strip()
    if x == '' or x.lower() == 'none':
        return 0.0
    try:
        if 'TB' in x:
            return float(x.replace('TB', '').strip()) * 1024
        elif 'GB' in x:
            return float(x.replace('GB', '').strip())
        return float(x)
    except ValueError:
        return 0.0


def cpu_category(x):
    """Bucket CPU strings into broad categories."""
    x = str(x)
    if 'Intel Core i7' in x:
        return 'Intel Core i7'
    elif 'Intel Core i5' in x:
        return 'Intel Core i5'
    elif 'Intel Core i3' in x:
        return 'Intel Core i3'
    elif 'Intel Celeron Dual Core N' in x:
        return 'Intel Celeron Dual Core N'
    elif 'Intel Pentium Quad Core N' in x:
        return 'Intel Pentium Quad Core N'
    elif 'Intel' in x:
        return 'Other Intel'
    elif 'AMD' in x:
        return 'All AMD'
    return 'Other'


def os_category(x):
    """Bucket OS strings into broad categories."""
    x = str(x)
    if 'Windows' in x:
        return 'Windows'
    elif 'Mac' in x or 'macOS' in x:
        return 'Mac'
    elif 'Linux' in x:
        return 'Linux'
    elif 'Chrome' in x:
        return 'Chrome'
    return 'Other'


def feature_engineering(df):
    """
    Transform raw laptop spec columns into model-ready features.

    Works for both batch (many rows) and single-row dataframes.
    Handles missing '+' in Memory column and missing IPS/Touchscreen
    text safely.
    """
    df = df.copy()

    # ---------------- RAM ----------------
    df['Ram'] = (
        df['Ram'].astype(str)
        .str.replace('GB', '', regex=False)
        .astype('int32')
    )

    # ---------------- Weight ----------------
    df['Weight'] = (
        df['Weight'].astype(str)
        .str.replace('kg', '', regex=False)
        .astype('float32')
    )

    # ---------------- Memory ----------------
    # FIX: previously, if NO row in the batch contained '+', memory_split
    # would have only 1 column and df['memory2'] would raise a KeyError.
    # We now always guarantee both columns exist.
    memory_split = df['Memory'].astype(str).str.split('+', expand=True)

    df['memory1'] = memory_split[0].str.strip()

    if memory_split.shape[1] > 1:
        df['memory2'] = memory_split[1].str.strip()
    else:
        df['memory2'] = np.nan

    split1 = df['memory1'].str.split(' ', n=1, expand=True)
    df['memory_size1'] = split1[0]
    df['memory_type1'] = split1[1] if split1.shape[1] > 1 else np.nan

    # FIX: same guarantee for the second memory slot
    temp = df['memory2'].astype(str).str.split(' ', n=1, expand=True)
    df['extra_memory_size'] = temp[0]
    if temp.shape[1] > 1:
        df['extra_memory_type'] = temp[1]
    else:
        df['extra_memory_type'] = np.nan
    # rows where memory2 was actually NaN -> extra_memory_size becomes
    # the string "nan"; convert_memory() handles that via pd.isna check
    # only for real NaN, so replace string "nan" explicitly
    df.loc[df['memory2'].isna(), 'extra_memory_size'] = np.nan
    df.loc[df['memory2'].isna(), 'extra_memory_type'] = np.nan

    df['Memory_size1'] = df['memory_size1'].apply(convert_memory)
    df['extra_memory'] = df['extra_memory_size'].apply(convert_memory)

    df['SSD'] = (
        np.where(df['memory_type1'] == 'SSD', df['Memory_size1'], 0)
        + np.where(df['extra_memory_type'] == 'SSD', df['extra_memory'], 0)
    )
    df['HDD'] = (
        np.where(df['memory_type1'] == 'HDD', df['Memory_size1'], 0)
        + np.where(df['extra_memory_type'] == 'HDD', df['extra_memory'], 0)
    )
    df['Flash'] = (
        np.where(df['memory_type1'] == 'Flash Storage', df['Memory_size1'], 0)
        + np.where(df['extra_memory_type'] == 'Flash Storage', df['extra_memory'], 0)
    )
    df['Hybrid'] = (
        np.where(df['memory_type1'] == 'Hybrid', df['Memory_size1'], 0)
        + np.where(df['extra_memory_type'] == 'Hybrid', df['extra_memory'], 0)
    )

    # ---------------- GPU ----------------
    df['Gpu_brand'] = df['Gpu'].astype(str).str.split().str[0]

    df['dedicated_gpu'] = np.where(
        df['Gpu_brand'].isin(['Nvidia', 'AMD']), 1, 0
    )

    df['gpu_model'] = pd.to_numeric(
        df['Gpu'].astype(str).str.extract(r'(\d{3,4})')[0],
        errors='coerce'
    )

    # ---------------- OS ----------------
    df['os_type'] = df['OpSys'].apply(os_category)

    # ---------------- CPU ----------------
    df['cpu_speed'] = (
        df['Cpu'].astype(str)
        .str.extract(r'(\d+\.?\d*)GHz')[0]
        .astype(float)
    )

    temp_cpu = (
        df['Cpu'].astype(str)
        .str.replace(r'(\d+\.?\d*GHz)', '', regex=True)
        .str.strip()
    )

    df['cpu_series'] = (
        temp_cpu
        .str.replace(r'(\d{4,}\w*)$', '', regex=True)
        .str.strip()
    )
    df['cpu_series'] = df['cpu_series'].apply(cpu_category)

    # ---------------- Screen ----------------
    screen_res = df['ScreenResolution'].astype(str)

    df['TouchScreen'] = screen_res.str.contains('Touchscreen').astype(int)
    df['IPS'] = screen_res.str.contains('IPS').astype(int)

    resolution = screen_res.str.split('x', expand=True)

    df['x_res'] = (
        resolution[0].str.split().str[-1].astype(int)
    )
    df['y_res'] = resolution[1].astype(int)

    df['Inches'] = df['Inches'].astype(float)

    df['ppi'] = (
        np.sqrt(df['x_res'] ** 2 + df['y_res'] ** 2) / df['Inches']
    )

    # ---------------- DROP ----------------
    drop_cols = [
        'Memory', 'memory1', 'memory2', 'memory_size1', 'memory_type1',
        'extra_memory_size', 'extra_memory_type', 'extra_memory',
        'Gpu', 'OpSys', 'Cpu', 'ScreenResolution',
        'x_res', 'y_res', 'Inches'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    return df


# Column groups used by the ColumnTransformer (after feature_engineering)
CAT_ATTRIBUTES = ['Company', 'TypeName', 'Gpu_brand', 'os_type', 'cpu_series']
BINARY_ATTRIBUTES = ['dedicated_gpu', 'TouchScreen', 'IPS']
GPU_MODEL_ATTRIBUTES = ['gpu_model']
ONLY_SCALE_ATTRIBUTES = ['Ram', 'cpu_speed']
NUM_ATTRIBUTES = ['Weight', 'SSD', 'HDD', 'Flash', 'Hybrid', 'ppi']


def build_pipeline():
    """Build the full sklearn Pipeline: feature engineering -> preprocessing -> model."""

    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    binary_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
    ])

    gpu_model_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1
        ))
    ])

    only_scale_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('scale', RobustScaler())
    ])

    num_pipeline = Pipeline([
        ('impute', KNNImputer(n_neighbors=5, weights='uniform', add_indicator=True)),
        ('power_transform', PowerTransformer(method='yeo-johnson')),
        ('scale', RobustScaler())
    ])

    preprocessing = ColumnTransformer([
        ('cat', cat_pipeline, CAT_ATTRIBUTES),
        ('binary', binary_pipeline, BINARY_ATTRIBUTES),
        ('gpu_model', gpu_model_pipeline, GPU_MODEL_ATTRIBUTES),
        ('only_scale', only_scale_pipeline, ONLY_SCALE_ATTRIBUTES),
        ('num', num_pipeline, NUM_ATTRIBUTES),
    ], remainder='drop')

    feature_transformer = FunctionTransformer(feature_engineering, validate=False)

    full_pipeline = Pipeline([
        ('feature_transformer', feature_transformer),
        ('preprocessing', preprocessing),
        ('model', LinearRegression())
    ])

    return full_pipeline
