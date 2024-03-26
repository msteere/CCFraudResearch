# https://www.researchgate.net/publication/374083997_FRAUD_DETECTION_USING_MACHINE_LEARNING

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def run_random_forest_classifier(file_path, target):
    data = pd.read_csv(file_path)
    X = data.drop(target, axis=1)
    y = data[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC Score: {roc_auc}")
    
    # Save the model if needed
    # joblib.dump(model, 'random_forest_model.pkl')
    
    return model.feature_importances_

def get_selected_features_from_random_forest(file_path, target, threshold=0.01):
    data = pd.read_csv(file_path)
    X = data.drop(target, axis=1)
    y = data[target]
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    feature_importances = model.feature_importances_
    
    # Now using the threshold to filter features
    selected_features = [feature for feature, importance in zip(X.columns, feature_importances) if importance >= threshold]
    
    print(f"Selected features based on Random Forest: {selected_features}")
    return selected_features

