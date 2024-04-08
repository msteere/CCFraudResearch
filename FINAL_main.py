import sys
from data_processing import preprocess_data
from FINAL_randomforest import run_rf
from FINAL_svm import run_svm
from FINAL_neuralnetwork import run_ann

def parse_command_line_arguments():
    file_path = sys.argv[1] 
    base_algorithm = sys.argv[2]
    scale_rule = sys.argv[3] # this could be none
    smote_rule = sys.argv[4] # this could be none
    enable_selected_features = sys.argv[5] # this should be True/False

    return file_path, base_algorithm, scale_rule, smote_rule, enable_selected_features

def train_and_run(base_algorithm, X_train, X_test, y_train, y_test):
    if base_algorithm is 'random_forest':
        # call with X_train, X_test, y_train, y_test
        run_rf(X_train, X_test, y_train, y_test)
    elif base_algorithm is 'svm':
        # call with X_train, X_test, y_train, y_test
        run_svm(X_train, X_test, y_train, y_test)
    elif base_algorithm is 'neural_network':
        kern_init = 'uniform' # NOTE: Change this
        run_ann(X_train, X_test, y_train, y_test, kern_init)
        # call with X_train, X_test, y_train, y_test, kern_init

if __name__ == "__main__":
    file_path, base_algorithm, scale_rule, smote_rule, enable_selected_features = parse_command_line_arguments()

    X_train, X_test, y_train, y_test = preprocess_data(file_path, scale_rule, smote_rule, enable_selected_features, target='Class')

    train_and_run(base_algorithm, X_train, X_test, y_train, y_test)

    # print_results()