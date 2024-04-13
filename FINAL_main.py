import sys
from data_processing import preprocess_data
#from FINAL_randomforest import run_rf
from FINAL_svm import run_svm_with_randomized_search_and_timeout
from FINAL_neuralnetwork import run_ann_with_timeout
from new_random_forest import rf_with_timeout
import random
import time


'''
   file_path = sys.argv[1] 
    base_algorithm = sys.argv[2] # Random forest, svm, or neural network
    scale_rule = sys.argv[3] # this could be none   THERE ARE 4 OF THESE: 'StandardScaler', 'RobustScaler', 'PowerTransformer', None
    smote_rule = sys.argv[4] # this could be none   3 OF THESE: 'smote', 'smote_enn', None
    enable_selected_features = sys.argv[5] # this should be True/False
'''






def random_parameters(parameters_grid):
    selected_parameters = {}
    for param, values in parameters_grid.items():
        selected_parameters[param] = random.choice(values)
    if(selected_parameters['base_algo']=='random_forest' and selected_parameters['enable_selected_features']):
        selected_parameters['enable_selected_features'] = 0
    return selected_parameters

def parse_command_line_arguments():
    file_path = sys.argv[1] 
    total_time = sys.argv[2]
    single_iter_time = sys.argv[3]
    #base_algorithm = sys.argv[2]
    #scale_rule = sys.argv[3] # this could be none
    #smote_rule = sys.argv[4] # this could be none
    #enable_selected_features = sys.argv[5] # this should be True/False

    return file_path, float(total_time), float(single_iter_time)#, base_algorithm, scale_rule, smote_rule, enable_selected_features

def train_and_run(file_path, base_algorithm, X_train, X_test, y_train, y_test, iter_start_time, single_iter_time):
    timeout = single_iter_time - (time.time()-iter_start_time)
    if (base_algorithm=='random_forest'):
        rf_with_timeout(file_path, X_train, X_test, y_train, y_test, timeout)
        # call with X_train, X_test, y_train, y_test
        #run_rf(X_train, X_test, y_train, y_test)
    elif (base_algorithm=='svm'):
        # call with X_train, X_test, y_train, y_test
        run_svm_with_randomized_search_and_timeout(X_train, X_test, y_train, y_test, single_iter_time, iter_start_time)
    elif (base_algorithm=='neural_network'):
        #kern_init = 'uniform' 
        run_ann_with_timeout(X_train, X_test, y_train, y_test, timeout)
        # call with X_train, X_test, y_train, y_test, kern_init


def total_grid_search(file_path, total_time, single_iter_time):
    start_time = time.time()
    parameters_grid = {
        # 'base_algo': ['random_forest', 'svm', 'neural_network'],
        'base_algo': ['svm', 'neural_network'],
        'scale_rule': ['StandardScaler', 'RobustScaler', 'PowerTransformer', None],
        'smote_rule' : ['smote', 'smote_enn', None],
        'enable_selected_features': [1, 0]
        # Add more parameters as needed
    }   
    while(1):
        iteration_start_time = time.time()
        if(time.time() - start_time > total_time):
            print('Total Grid Search time limit reached.\n')
            return
        selected_parameters = random_parameters(parameters_grid)
        base_algorithm = selected_parameters['base_algo']
        scale_rule = selected_parameters['scale_rule']
        smote_rule = selected_parameters['smote_rule']
        enable_selected_features = selected_parameters['enable_selected_features']

        rf = ''
        if(enable_selected_features):
            rf = ' random forest feature selection,'
        else:
            rf = 'out random forest feature selection,'

        print('Preprocessing data for base_algorithm= ', base_algorithm, ' with', rf, 'scale_rule=', scale_rule, ' smote_rule=', smote_rule, '\n')

        # NOTE: Here we are trying to adjust the process for selected_features so that it's preset and hardcoded.
        X_train, X_test, y_train, y_test = preprocess_data(file_path, scale_rule, smote_rule, enable_selected_features, target='Class')

        train_and_run(file_path, base_algorithm, X_train, X_test, y_train, y_test, iteration_start_time, single_iter_time)




if __name__ == "__main__":
    file_path, total_time, single_iter_time = parse_command_line_arguments()
    print("Total time: ", total_time, "\niteration time: ", single_iter_time, "\n")
    total_grid_search(file_path, total_time, single_iter_time)
    