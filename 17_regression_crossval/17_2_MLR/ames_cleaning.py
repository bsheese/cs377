import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def load_and_clean_ames(url, test_size=0.2, random_state=42):
    # Example usage:
    # url = 'https://raw.githubusercontent.com/bsheese/CSDS125ExampleData/master/data_housing_ames.txt'
    # X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = load_and_clean_ames(url)
    # print(f"Cleaned X_train shape: {X_train_cleaned.shape}")
    # print(f"Cleaned X_test shape: {X_test_cleaned.shape}")
    
    # Data Loading and Initial Cleaning
    df = pd.read_csv(url, sep='\t')

    # Drop as recommended by the author of the dataset (unusual sales)
    df = df[df["Gr Liv Area"] < 4000]

    # Drop Identifiers.
    df = df.drop(['Order', 'PID'], axis=1)

    # Drop near-constant or highly sparse features to keep the dataset manageable for beginners
    sparse_drops = ['Pool QC', 'Pool Area', 'Misc Feature', 'Misc Val', 'Alley', 'Fence']
    constant_drops = ['Street', 'Utilities', 'Condition 2', 'Roof Matl', 'Heating', 'Low Qual Fin SF', '3Ssn Porch']

    # Drop 'Garage Yr Blt' to simplify and avoid leakage
    df = df.drop(sparse_drops + constant_drops + ['Garage Yr Blt'], axis=1, errors='ignore')

    # Correct Data Types: MSSubClass is a numeric code, not a magnitude. Convert to string.
    df['MS SubClass'] = df['MS SubClass'].astype(str)

    # Resolve "Meaningful" NAs (Part 1 of Missing Values handling)
    # Basement NAs mean "No Basement"
    bsmt_cols = ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
    for col in bsmt_cols:
        df[col] = df[col].fillna('None')

    # Garage NAs mean "No Garage"
    garage_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
    for col in garage_cols:
        df[col] = df[col].fillna('None')

    # Fireplace NAs mean "No Fireplace"
    df['Fireplace Qu'] = df['Fireplace Qu'].fillna('None')

    # Log Transformations for skewed columns
    skewed_cols = ['Lot Area', 'Gr Liv Area', 'Mas Vnr Area', 'SalePrice']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for col in skewed_cols:
            if col == 'Mas Vnr Area':
                df['Log_'+ col] = np.log1p(df[col]) # this is plus 1 then add the log
            else:
                df['Log_'+ col] = np.log(df[col])
            df = df.drop(columns = col)

    # Train/Test Split
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    # Statistical Imputation (median replacement) - applied after split
    numeric_cols = df_train.select_dtypes(include=np.number).columns
    imputer = SimpleImputer(strategy='median')
    df_train[numeric_cols] = imputer.fit_transform(df_train[numeric_cols])
    df_test[numeric_cols] = imputer.transform(df_test[numeric_cols])

    # Feature Engineering
    df_train['Total_Square_Footage'] = df_train['1st Flr SF'] + df_train['2nd Flr SF'] + df_train['Total Bsmt SF']
    df_test['Total_Square_Footage'] = df_test['1st Flr SF'] + df_test['2nd Flr SF'] + df_test['Total Bsmt SF']
    df_train = df_train.drop(['1st Flr SF', '2nd Flr SF', 'Total Bsmt SF'], axis=1)
    df_test = df_test.drop(['1st Flr SF', '2nd Flr SF', 'Total Bsmt SF'], axis=1)

    # Ordinal Encoding
    quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    ordinal_cols = ['Exter Qual', 'Kitchen Qual', 'Heating QC', 'Bsmt Qual', 'Fireplace Qu']

    for col in ordinal_cols:
        df_train[col] = df_train[col].map(quality_map).fillna(3)
        df_test[col] = df_test[col].map(quality_map).fillna(3)

    # Garage Finish has a different mapping
    garage_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}
    df_train['Garage Finish'] = df_train['Garage Finish'].map(garage_map).fillna(0)
    df_test['Garage Finish'] = df_test['Garage Finish'].map(garage_map).fillna(0)

    # Nominal Encoding (One-Hot)
    nominal_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    encoded_train = encoder.fit_transform(df_train[nominal_cols])
    encoded_test = encoder.transform(df_test[nominal_cols])
    encoded_cols = encoder.get_feature_names_out(nominal_cols)
    df_encoded_train = pd.DataFrame(encoded_train, columns=encoded_cols, index=df_train.index)
    df_encoded_test = pd.DataFrame(encoded_test, columns=encoded_cols, index=df_test.index)
    df_train = df_train.drop(nominal_cols, axis=1).join(df_encoded_train)
    df_test = df_test.drop(nominal_cols, axis=1).join(df_encoded_test)

    # Separate Features (X) and Target (y)
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
