# https://www.researchgate.net/publication/374083997_FRAUD_DETECTION_USING_MACHINE_LEARNING

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def run_rf(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC Score: {roc_auc}")
    
    # Save the model if needed
    # joblib.dump(model, 'random_forest_model.pkl')
    
    return model.feature_importances_

def get_selected_features_from_random_forest(X_train, y_train, threshold=0.01):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    
    # Now using the threshold to filter features
    selected_features = [feature for feature, importance in zip(X_train.columns, feature_importances) if importance >= threshold]
    
    print(f"Selected features based on Random Forest: {selected_features}")
    return selected_features