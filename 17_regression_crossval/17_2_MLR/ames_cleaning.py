import pandas as pd
import numpy as np

def safe_drop(df: pd.DataFrame, drop_list: list[str]) -> pd.DataFrame:
    """Safely drops columns from a DataFrame if they exist."""
    existing_cols_to_drop = [col for col in drop_list if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(existing_cols_to_drop, axis=1)
    return df


def load_and_clean_ames(
    url: str = "https://raw.githubusercontent.com/bsheese/CSDS125ExampleData/master/data_housing_ames.txt",
    drop_outliers: bool = True,
    house_area_threshold: int = 4000,
    fix_garage_year: bool = True,
    add_garage_features: bool = True,
    one_hot_encode: bool = False
) -> pd.DataFrame:
    """
    Load and clean the Ames Housing dataset.
    
    Args:
        url: URL to the Ames Housing data file
        drop_outliers: Whether to remove houses with Gr Liv Area > threshold
        house_area_threshold: Maximum living area to keep (default 4000)
        fix_garage_year: Whether to cap garage year built at 2010
        add_garage_features: Whether to create garage_attached and garage_finished flags
        one_hot_encode: Whether to one-hot encode categorical variables
    
    Returns:
        Cleaned DataFrame ready for modeling
    """
    # Load data
    df = pd.read_csv(url, sep='\t')
    
    # Remove outliers (large houses sold for little due to inheritance)
    if drop_outliers:
        df = df.loc[df['Gr Liv Area'] < house_area_threshold, :].copy()
    
    # Fix future garage year built
    if fix_garage_year and 'Garage Yr Blt' in df.columns:
        df.loc[df['Garage Yr Blt'] > 2010, 'Garage Yr Blt'] = 2010
    
    # Create garage features
    if add_garage_features:
        if 'Garage Type' in df.columns:
            df['garage_attached'] = np.where(df['Garage Type'] == 'Attchd', 1, 0)
        if 'Garage Finish' in df.columns:
            df['garage_finished'] = np.where(df['Garage Finish'] != 'Unf', 1, 0)
            df['garage_unfinished'] = np.where(df['Garage Finish'] == 'Unf', 1, 0)
    
    # Drop list of uninformative columns
    drop_list = [
        'Order', 'PID',
        'Pool QC', 'Pool Area', 'Misc Feature', 'Misc Val',
        'Alley', 'Fence', 'Mas Vnr Type',
        'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2',
        'Fireplace Qu',
        'Neighborhood', 'MS Subclass', 'Mo Sold',
        'Kitchen Qual', 'Exter Qual', 'Heating QC',
        'Garage Qual', 'Garage Cond', 'Garage Type', 'Garage Finish',
        'Street', 'Land Contour', 'Utilities', 'Land Slope',
        'Condition 1', 'Condition 2', 'Roof Matl', 'Exter Cond',
        'Heating', 'Central Air', 'Electrical', 'Functional',
        'Paved Drive', 'Sale Type',
        'Exterior 1st', 'Exterior 2nd',
        'Mas Vnr Area'
    ]
    df = safe_drop(df, drop_list)
    
    # Collapse high-cardinality categoricals (>50% single value to binary flag)
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].value_counts(normalize=True, dropna=False).max() > 0.50:
            top_value = df[col].value_counts(normalize=True, dropna=False).index[0]
            df[col + '_' + top_value] = np.where(df[col] == top_value, 1, 0)
            df = safe_drop(df, [col])
    
    # Handle Foundation - collapse rare categories
    if 'Foundation' in df.columns:
        df.loc[~df['Foundation'].isin(['PConc', 'CBlock']), 'Foundation'] = 'Other'
    
    # Convert object columns to category dtype
    for col in df.select_dtypes('object').columns:
        df[col] = df[col].astype('category')
    
    # Convert binary columns to bool
    for col in df.columns:
        if df[col].value_counts().shape[0] == 2:
            df[col] = df[col].astype('bool')
    
    # Drop highly uniform numeric columns (>90% single value)
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].value_counts(normalize=True, dropna=False).max() > 0.90:
            df = safe_drop(df, [col])
    
    # Convert near-zero numerics to binary (>80% single value, >2 unique values)
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].value_counts(normalize=True, dropna=False).max() > 0.80:
            if len(df[col].unique()) > 2:
                df[col] = np.where(df[col] > 0, 1, 0)
            df[col] = df[col].astype('boolean')
    
    # Fill numeric NaN with median
    num_cols_with_na = df.select_dtypes(include=np.number).columns[
        df.select_dtypes(include=np.number).isnull().any()
    ].tolist()
    for col in num_cols_with_na:
        df[col] = df[col].fillna(df[col].median())
    
    # Optional: One-hot encode remaining categoricals
    if one_hot_encode:
        df = pd.get_dummies(df, columns=df.select_dtypes(include='category').columns, drop_first=True)
    else:
        # Convert category columns to numeric codes for tree models
        for col in df.select_dtypes(include='category').columns:
            df[col] = df[col].cat.codes
    
    return df


def get_colab_download_code(module_url: str = None) -> str:
    """Return the code needed to download and import this module in Colab."""
    if module_url is None:
        module_url = "https://raw.githubusercontent.com/bsheese/cs377/main/17_regression_crossval/ames_cleaning.py"
    return f'''import urllib.request
module_url = "{module_url}"
urllib.request.urlretrieve(module_url, "ames_cleaning.py")
from ames_cleaning import load_and_clean_ames'''