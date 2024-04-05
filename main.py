from random_forest import run_random_forest_classifier, get_selected_features_from_random_forest
from svm import run_svm_with_all_features, run_svm_with_selected_features
from svm_graph_search import run_svm_with_randomized_search_and_timeout
# from neural_network import run_ann
from neural_network2 import run_ann
# from neural_network3 import run_ann_with_adaboost_concept
# from neural_network4 import run_ann_with_adaboost
from selected_features import SELECTED_FEATURES
# Other necessary imports

def main():
    # Load the datasets
    file_path = 'creditcard.csv'  # Path to untouched data
    file_path_normalized = 'creditcard_normalized.csv'  # Path to normalized data
    # data = load_data(file_path)
    # data_normalized = load_data(file_path_normalized)
    target = 'Class'

    # Present the choices
    print("Select an algorithm to run:")
    print("0) Run Random Forest classifier")
    print("1) Random Forest Preprocessing (selecting most important features from feature set)")
    print("2) SVM with all features")
    print("3) SVM with only features from Random Forest preprocessing")
    print("4) SVM with normalized data and all features")
    print("5) SVM with only features from Random Forest preprocessing and normalized data")
    print("6) Neural Network with all features")
    print("7) Neural Network with only features from Random Forest preprocessing")
    print("8) Neural Network with normalized data and all features")
    print("9) Neural Network with only features from Random Forest preprocessing and normalized data")
    
    # Get user choice
    choice = int(input("Enter the number of your choice: "))

    if choice == 0:
        results = run_random_forest_classifier(file_path, target)
    elif choice == 1:
        selected_features = get_selected_features_from_random_forest(file_path, target)
        print(f"Selected features are: {selected_features}")
    elif choice == 2:
        results = run_svm_with_all_features(file_path, target)
    elif choice == 3:
        results = run_svm_with_selected_features(file_path, SELECTED_FEATURES, target)
    elif choice == 4:
        results = run_svm_with_all_features(file_path_normalized, target)
    elif choice == 5:
        results = run_svm_with_selected_features(file_path_normalized, SELECTED_FEATURES, target)
    elif choice == 6:
        results = run_neural_network_with_all_features(file_path, target)
    elif choice == 7:
        results = run_neural_network_with_selected_features(file_path, SELECTED_FEATURES, target)
    elif choice == 8:
        results = run_neural_network_with_all_features(file_path_normalized, target)
    elif choice == 9:
        results = run_neural_network_with_selected_features(file_path_normalized, SELECTED_FEATURES, target)
    else:
        print("Invalid choice. Please enter a number between 0 and 9.")
        return

    

if __name__ == "__main__":
    
    # NOTE: Add validation as part of training set
    # NOTE: Might want to try scaler types:
        # RobustScaler for any option
        # PowerTransformer for neural network
    # NOTE: Could consider implementing XGBoost or something into RF &/or SVM if time
    
    # OPTION 1: RF
        # Normal
        # Normal with smote vs. smote_enn
        # Normal with Selected features
    
        # Scaled
        # Scaled with smote vs. smote_enn
        # Scaled with Selected features 
    
    # OPTION 2: SVM
        # Normal
        # Normal with smote vs. smote_enn
        # Normal with Selected features
    
        # Scaled
        # Scaled with smote vs. smote_enn
        # Scaled with Selected features
    
    # OPTION 3: Neural Network
        # Normal
        # Normal with smote vs. smote_enn
        # Normal with Selected features
    
        # Scaled
        # Scaled with smote vs. smote_enn
        # Scaled with Selected features
    
        # Number of layers
        # Units per layer
        # Internal Activation = 'relu' vs. LeakyRelu vs. 'elu' vs. 'selu'
            # Might wanna go from 'relu' to Leaky or elu or selu
        # kernal_initializer = Uniform vs. Normal vs. HeNormal()
        # Optimizer = 'adam' vs. 'Nadam' vs. 'RMSprop'
        # Epochs <---- This can have an effect on overfitting if too large
    
    
    
    
    # main()
    # run_svm_with_randomized_search_and_timeout('creditcard_normalized.csv', search_time_limit=5500, time_limit_for_iteration=1000, n_iter=10)
    # run_ann('creditcard.csv', 'Class')
    # run_svm_with_selected_features('creditcard_normalized.csv', SELECTED_FEATURES,'Class')
    # run_ann('creditcard_normalized.csv', 'Class', SELECTED_FEATURES)

    # Adjust file path, target, and selected_features as needed
    # run_ann_with_adaboost_concept('creditcard_normalized.csv', 'Class', selected_features=None, n_models=5)

    #run_ann_with_adaboost('creditcard.csv')

    run_ann('creditcard.csv')


