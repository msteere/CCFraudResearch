
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def run_svm_with_all_features(file_path, target='Class'):
    data = pd.read_csv(file_path)
    X = data.drop(target, axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)  # Example: optimized parameters
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

def run_svm_with_selected_features(file_path_normalized, selected_features, target='Class'):
    data = pd.read_csv(file_path_normalized)
    # Ensure 'Class' is part of selected_features for proper data slicing
    data = data[selected_features + [target]] if target not in selected_features else data[selected_features]
    X = data.drop(target, axis=1)
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)  # Assuming these parameters are optimized
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
