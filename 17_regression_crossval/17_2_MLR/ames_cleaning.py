import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_clean_ames(
    url: str = "https://raw.githubusercontent.com/bsheese/CSDS125ExampleData/master/data_housing_ames.txt",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Load, clean, and split the Ames Housing dataset following MLR best practices.
    Returns: X_train, X_test, y_train, y_test
    """
    # Load data
    df = pd.read_csv(url, sep='\t')
    
    # 1. Drop Identifiers
    df = df.drop(['Order', 'PID'], axis=1, errors='ignore')
    
    # 2. Correct Data Types
    df['MS SubClass'] = df['MS SubClass'].astype(str)
    
    # 3. Resolve Meaningful NAs
    bsmt_cols = ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
    for col in bsmt_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
            
    garage_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
    for col in garage_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
            
    if 'Fireplace Qu' in df.columns:
        df['Fireplace Qu'] = df['Fireplace Qu'].fillna('None')

    # Drop uninformative columns to keep things simple
    sparse_drops = ['Pool QC', 'Pool Area', 'Misc Feature', 'Misc Val', 'Alley', 'Fence']
    constant_drops = ['Street', 'Utilities', 'Condition 2', 'Roof Matl', 'Heating', 'Low Qual Fin SF', '3Ssn Porch']
    df = df.drop(sparse_drops + constant_drops + ['Garage Yr Blt'], axis=1, errors='ignore')
    
    # 4. Train/Test Split
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # 5. Remove Extreme Outliers (Train set only)
    df_train = df_train[df_train['Gr Liv Area'] <= 4000].copy()
    
    # 6. Statistical Imputation
    numeric_cols = df_train.select_dtypes(include=np.number).columns
    cols_with_na = numeric_cols[df_train[numeric_cols].isna().any()].tolist()
    
    for col in cols_with_na:
        median_val = df_train[col].median()
        df_train[col] = df_train[col].fillna(median_val)
        df_test[col] = df_test[col].fillna(median_val)
        
    # 7. Transform Skewed Variables
    df_train['Log_SalePrice'] = np.log(df_train['SalePrice'])
    df_test['Log_SalePrice'] = np.log(df_test['SalePrice'])
    
    df_train = df_train.drop('SalePrice', axis=1)
    df_test = df_test.drop('SalePrice', axis=1)
    
    # 8. Create Synergistic Features
    df_train['Total_Square_Footage'] = df_train['1st Flr SF'] + df_train['2nd Flr SF'] + df_train['Total Bsmt SF']
    df_test['Total_Square_Footage'] = df_test['1st Flr SF'] + df_test['2nd Flr SF'] + df_test['Total Bsmt SF']
    
    df_train = df_train.drop(['1st Flr SF', '2nd Flr SF', 'Total Bsmt SF'], axis=1)
    df_test = df_test.drop(['1st Flr SF', '2nd Flr SF', 'Total Bsmt SF'], axis=1)
    
    # 9. Ordinal Encoding
    quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    ordinal_cols = ['Exter Qual', 'Kitchen Qual', 'Heating QC', 'Bsmt Qual', 'Fireplace Qu']
    
    for col in ordinal_cols:
        if col in df_train.columns:
            df_train[col] = df_train[col].map(quality_map).fillna(3)
            df_test[col] = df_test[col].map(quality_map).fillna(3)
            
    garage_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}
    if 'Garage Finish' in df_train.columns:
        df_train['Garage Finish'] = df_train['Garage Finish'].map(garage_map).fillna(0)
        df_test['Garage Finish'] = df_test['Garage Finish'].map(garage_map).fillna(0)
        
    # 10. Nominal Encoding (One-Hot)
    nominal_cols = ['Neighborhood', 'Foundation', 'MS Zoning', 'Bldg Type', 'Central Air']
    existing_nominal = [c for c in nominal_cols if c in df_train.columns]
    
    df_combined = pd.concat([df_train, df_test])
    df_combined = pd.get_dummies(df_combined, columns=existing_nominal, drop_first=True)
    
    objects_to_drop = df_combined.select_dtypes(include='object').columns
    df_combined = df_combined.drop(objects_to_drop, axis=1)
    
    df_train = df_combined.loc[df_train.index].copy()
    df_test = df_combined.loc[df_test.index].copy()
    
    # 11. Multicollinearity Drop
    if 'Garage Area' in df_train.columns and 'Garage Cars' in df_train.columns:
        df_train = df_train.drop('Garage Area', axis=1)
        df_test = df_test.drop('Garage Area', axis=1)
        
    # Split into X and y
    X_train = df_train.drop('Log_SalePrice', axis=1)
    y_train = df_train['Log_SalePrice']
    X_test = df_test.drop('Log_SalePrice', axis=1)
    y_test = df_test['Log_SalePrice']
    
    return X_train, X_test, y_train, y_test

def get_colab_download_code(module_url: str = None) -> str:
    """Return the code needed to download and import this module in Colab."""
    if module_url is None:
        module_url = "https://raw.githubusercontent.com/bsheese/cs377/main/17_regression_crossval/17_2_MLR/ames_cleaning.py"
    return f'''import urllib.request
module_url = "{module_url}"
urllib.request.urlretrieve(module_url, "ames_cleaning.py")
from ames_cleaning import load_and_clean_ames'''
