from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.stats import expon
import concurrent.futures
import time

def svm_train_with_timeout(X_train, y_train, X_test, y_test, params, timeout):
    """Train and evaluate an SVM model with a timeout."""
    def train_and_evaluate():
        model = SVC(probability=True, **params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        return classification_report(y_test, predictions), accuracy_score(y_test, predictions), auc_score

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(train_and_evaluate)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"Training exceeded time limit of {timeout} seconds for parameters: {params}")
            return None, None, None

def run_svm_with_randomized_search_and_timeout(file_path, target='Class', search_time_limit=600, time_limit_for_iteration=300, n_iter=10):
    data = pd.read_csv(file_path)
    X = data.drop(target, axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_distributions = {
    'C': np.logspace(-3, 2, 6),  # Generates values [0.001, 0.01, 0.1, 1, 10, 100]
    'gamma': np.logspace(-3, -1, 3),  # Generates values [0.001, 0.01, 0.1]
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']  # Kernel types
}

    param_sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=42)

    start_time = time.time()
    for params in param_sampler:
        iteration_start_time = time.time()
        if time.time() - start_time > search_time_limit:
            print("Search time limit exceeded.")
            return
        if(time.time()-iteration_start_time > time_limit_for_iteration):
            print("Time limit for specific set of parameters reached")

        print(f"Testing parameters: {params}")
        timeout_arg = min(search_time_limit - (time.time() - start_time), time_limit_for_iteration)
        report, accuracy, auc = svm_train_with_timeout(X_train, y_train, X_test, y_test, params, timeout=timeout_arg)
        if report is not None:
            print(report)
            print(f"Accuracy: {accuracy}")
            print(f"AUC: {auc}")
        print("--------")

# Example usage
# Adjust file_path to your dataset's path
# run_svm_with_randomized_search_and_timeout('path/to/your/dataset.csv', search_time_limit=600, n_iter=10)
