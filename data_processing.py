import panda as pd

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split

from selected_features import SELECTED_FEATURES

def get_data(file_path, standardization_rule, smote, enable_selected_features, target='Class'):
    

    # ENABLE SELECTED FEATURES
    

    X = data.drop(target, axis=1)
    y = data[target].values

    
    
    # SMOTE
    if smote == 'smote':
        #
    elif smote == 'smote_enn'
        #

def scale(scale_rule, X):
    if scale_rule == 'StandardScaler':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scale_rule == 'RobustScaler':
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    elif scale_rule == 'PowerTransformer':
        scaler = PowerTransformer()
        X_scaled = scaler.fit_transform(X)

    return X

def apply_smote(smote_rule, X_train, y_train):
    if smote_rule == 'smote':
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    elif smote_rule == 'smote_enn':
        smote_enn = SMOTEENN(random_state=42)
        X_train, y_train = smote_enn.fit_resample(X_train, y_train)

    return X_train, y_train


def preprocess_data(file_path, scale_rule, smote_rule, enable_selected_features, target='Class'):
    data = pd.read_csv(file_path)
    
    if enable_selected_features:
        features = SELECTED_FEATURES + [target] if target not in SELECTED_FEATURES else SELECTED_FEATURES
        data = data[features]

    X = data.drop(target, axis=1)
    y = data[target].values

    X = data.scale(scale_rule, X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # NOTE: Add validation somehow right here

    X_train, y_train = apply_smote(smote_rule, X_train, y_train)
    
    return X_train, X_test, y_train, y_test