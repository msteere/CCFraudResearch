import sys
from data_processing import preprocess_data

def parse_command_line_arguments():
    file_path = sys.argv[1] 
    base_algorithm = sys.argv[2]
    scale_rule = sys.argv[3] # this could be none
    smote_rule = sys.argv[4] # this could be none
    enable_selected_features = sys.argv[5] # this should be True/False

    return file_path, base_algorithm, scale_rule, smote_rule, enable_selected_features

def train_and_run(base_algorithm, X_train, X_test, y_train, y_test):
    if base_algorithm == 'random_forest':
        #
    elif base_algorithm == 'svm':
        #
    elif base_algorithm == 'neural_network':
        #

if __name__ == "__main__":
    file_path, base_algorithm, scale_rule, smote_rule, enable_selected_features = parse_command_line_arguments()

    X_train, X_test, y_train, y_test = preprocess_data(file_path, scale_rule, smote_rule, enable_selected_features, target='Class')

    train_and_run(base_algorithm, X_train, X_test, y_train, y_test)

    # print_results()



