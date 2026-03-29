import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_clean_titanic(
    url: str = "https://raw.githubusercontent.com/bsheese/CSDS125ExampleData/master/data_titanic.csv",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Load and clean the Titanic dataset.
    
    Args:
        url: URL to the Titanic CSV file
        test_size: Proportion for test split
        random_state: Seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test: Train-test split features and target
        features: List of feature names used
    """
    # Load data
    df = pd.read_csv(url)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' aboard', '', regex=False)
    
    # Create binary indicators for family presence
    df['sibspouse'] = (df['siblings/spouses'] > 0).astype(int)
    df['parentchild'] = (df['parents/children'] > 0).astype(int)
    
    # Encode sex: male=0, female=1
    df['sex'] = (df['sex'] == 'female').astype(int)
    
    # Log-transform fare to handle skewness
    df['fare_log'] = np.log1p(df['fare'])
    
    # Handle missing age values - fill with median
    df['age'] = df['age'].fillna(df['age'].median())
    
    # Drop original columns we won't use
    df = df.drop(columns=['fare', 'name', 'ticket', 'cabin', 'embarked'], errors='ignore')
    
    # One-hot encode passenger class
    df = pd.get_dummies(df, columns=['pclass'], prefix='pclass', drop_first=False)
    
    # Define features and target
    features = ['age', 'sibspouse', 'parentchild', 'fare_log', 'sex', 'pclass_1', 'pclass_2', 'pclass_3']
    
    X = df[features]
    y = df['survived']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, features


def load_breast_cancer_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load and prepare the Wisconsin Breast Cancer dataset.
    
    Args:
        test_size: Proportion for test split
        random_state: Seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test: Train-test split features and target
        feature_names: List of feature names
    """
    from sklearn.datasets import load_breast_cancer
    
    # Load data
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    
    # Separate features and target
    X = df[data.target_names]
    y = data.target
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, list(X.columns)


def get_colab_download_code(module_url: str = None) -> str:
    """Return the code needed to download and import this module in Colab."""
    if module_url is None:
        module_url = "https://raw.githubusercontent.com/bsheese/cs377/main/18_classification/classification_cleaning.py"
    return f'''import urllib.request
module_url = "{module_url}"
urllib.request.urlretrieve(module_url, "classification_cleaning.py")
from classification_cleaning import load_and_clean_titanic, load_breast_cancer_data'''