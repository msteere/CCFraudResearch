import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, recall_score, roc_auc_score
from imblearn.combine import SMOTEENN  # If using imbalanced-learn for SMOTEENN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam  # Example if you need a custom optimizer
# Ensure you have imblearn installed: pip install imbalanced-learn


# Assuming your neural network model is defined as 'NeuralNetworkModel'
# and you've prepared your data in X_train, X_test, y_train, y_test

class AdaBoostClassifier(object):
    #def __init__(self, base_estimator, n_estimators=50, learning_rate=1.0):
        #self.base_estimator = base_estimator
        #self.n_estimators = n_estimators
        #self.learning_rate = learning_rate
        # Initialize additional necessary attributes here

    def __init__(self, base_estimator, n_estimators=30, learning_rate=1.0, algorithm='SAMME.R', random_state=42, epochs=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm_ = algorithm  # Initialize algorithm_ attribute based on the algorithm parameter
        self.random_state_ = random_state
        self.epochs=10

        # Initialize the list of estimators, their weights, and errors
        self.estimators_ = []  # List to store base estimators
        self.estimator_weights_ = np.zeros(self.n_estimators)  # Array to store estimator weights
        self.estimator_errors_ = np.zeros(self.n_estimators)  # Array to store estimator errors

    def _samme_proba(self, estimator, n_classes, X):
        """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

        """
        proba = estimator.predict(X)

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)

        return (n_classes - 1) * (log_proba - (1. / n_classes)
                                  * log_proba.sum(axis=1)[:, np.newaxis])
    
    def fit(self, X, y, batch_size):
        
        ## CNN:
        self.batch_size = batch_size
        
        #        self.epochs = epochs
        self.n_samples = X.shape[0]
        # There is hidden trouble for classes, here the classes will be sorted.
        # So in boost we have to ensure that the predict results have the same classes sort
        
        self.classes_ = np.array(sorted(list(set(y))))
        
        ############for CNN (2):
        #        yl = np.argmax(y)
        #        self.classes_ = np.array(sorted(list(set(yl))))

        self.n_classes_ = len(self.classes_)
        for iboost in range(self.n_estimators):
            if iboost == 0:
                sample_weight = np.ones(self.n_samples) / self.n_samples

            sample_weight, estimator_weight, estimator_error = self.boost(X, y, sample_weight)

            # early stop
            if estimator_error == None:
                break

            # append error and weight
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight

            if estimator_error <= 0:
                break

        return self
    
    def boost(self, X, y, sample_weight):
        if self.algorithm_ == 'SAMME':
            return self.discrete_boost(X, y, sample_weight)
        elif self.algorithm_ == 'SAMME.R':
            return self.real_boost(X, y, sample_weight)
        

    def real_boost(self, X, y, sample_weight):
        #            estimator = deepcopy(self.base_estimator_)
        ############################################### my code:
          
        if len(self.estimators_) == 0:
            #Copy CNN to estimator:
            estimator = self.deepcopy_CNN(self.base_estimator)#deepcopy of self.base_estimator_
        else: 
            #estimator = deepcopy(self.estimators_[-1])
            estimator = self.deepcopy_CNN(self.estimators_[-1])#deepcopy CNN
        ###################################################
        #if self.random_state_:
        #        estimator.set_params(random_state=1)
        #        estimator.fit(X, y, sample_weight=sample_weight)
        #################################### CNN (3) binery label:       
        # lb=LabelBinarizer()
        # y_b = lb.fit_transform(y)
        
        #lb=OneHotEncoder(sparse=False)
        #y_b=y.reshape(len(y),1)
        #y_b=lb.fit_transform(y_b)
        
        estimator.fit(X, y, sample_weight=sample_weight, epochs = self.epochs, batch_size = self.batch_size)
        ############################################################
        y_pred = estimator.predict(X)
        ############################################ (4) CNN :
        y_pred_l = np.argmax(y_pred, axis=1)
        incorrect = y_pred_l != y
        #########################################################        
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # if worse than random guess, stop boosting
        if estimator_error >= 1.0 - 1 / self.n_classes_:
            return None, None, None

        y_predict_proba = estimator.predict(X)
 
        # repalce zero
        y_predict_proba[y_predict_proba < np.finfo(y_predict_proba.dtype).eps] = np.finfo(y_predict_proba.dtype).eps

        y_codes = np.array([-1. / (self.n_classes_ - 1), 1.])
        y_coding = y_codes.take(self.classes_ == y[:, np.newaxis])

        # for sample weight update
        intermediate_variable = (-1. * self.learning_rate_ * (((self.n_classes_ - 1) / self.n_classes_) *
                                                              inner1d(y_coding, np.log(
                                                                  y_predict_proba))))  #dot iterate for each row

        # update sample weight
        sample_weight *= np.exp(intermediate_variable)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # normalize sample weight
        sample_weight /= sample_weight_sum

        # append the estimator
        self.estimators_.append(estimator)

        return sample_weight, 1, estimator_error
    
    def deepcopy_CNN(self, base_estimator0):
        #Copy CNN (self.base_estimator_) to estimator:
        config=base_estimator0.get_config()
        #estimator = Models.model_from_config(config)
        estimator = Sequential.from_config(config)

        
        weights = base_estimator0.get_weights()
        estimator.set_weights(weights)
        # estimator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        estimator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return estimator 
    
    def discrete_boost(self, X, y, sample_weight):
#        estimator = deepcopy(self.base_estimator_)
         ############################################### my code:
           
        if len(self.estimators_) == 0:
            #Copy CNN to estimator:
            estimator = self.deepcopy_CNN(self.base_estimator)#deepcopy of self.base_estimator_
        else: 
            #estimator = deepcopy(self.estimators_[-1])
            estimator = self.deepcopy_CNN(self.estimators_[-1])#deepcopy CNN
            ###################################################
        
        if self.random_state_:
            estimator.set_params(random_state=1)
            #        estimator.fit(X, y, sample_weight=sample_weight)
            #################################### CNN (3) binery label:       
            # lb=LabelBinarizer()
            # y_b = lb.fit_transform(y)
        
        #lb=OneHotEncoder(sparse=False)
        #y_b=y.reshape(len(y),1)
        #y_b=lb.fit_transform(y_b)
        
        estimator.fit(X, y, sample_weight=sample_weight, epochs = self.epochs, batch_size = self.batch_size)
        ############################################################        
        y_pred = estimator.predict(X)
        
        #incorrect = y_pred != y
        ############################################ (4) CNN :
        y_pred_l = np.argmax(y_pred, axis=1)
        incorrect = y_pred_l != y
        #######################################################   
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # if worse than random guess, stop boosting
        if estimator_error >= 1 - 1 / self.n_classes_:
            return None, None, None

        # update estimator_weight
        #        estimator_weight = self.learning_rate_ * np.log((1 - estimator_error) / estimator_error) + np.log(
        #            self.n_classes_ - 1)
        estimator_weight = self.learning_rate_ * (np.log((1. - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1.))

        if estimator_weight <= 0:
            return None, None, None

        # update sample weight
        sample_weight *= np.exp(estimator_weight * incorrect)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # normalize sample weight
        sample_weight /= sample_weight_sum

        # append the estimator
        self.estimators_.append(estimator)

        return sample_weight, estimator_weight, estimator_error
    
    def predict(self, X):
        if not self.estimators_:
            raise ValueError("No estimators trained!")

            # Ensure the sum of weights is not zero
        if np.isclose(self.estimator_weights_.sum(), 0):
            raise ValueError("Sum of estimator weights is zero. Can't make predictions.")

        # Aggregate predictions
        weighted_predictions = np.zeros((X.shape[0], len(self.estimators_)))
        for idx, estimator in enumerate(self.estimators_):
            predictions = estimator.predict(X)
            weighted_predictions[:, idx] = predictions.ravel() * self.estimator_weights_[idx]

        # Sum weighted predictions and determine final output class
        aggregated_predictions = np.sum(weighted_predictions, axis=1)
        y_pred_classes = (aggregated_predictions > 0).astype(int)  # Using 0 as threshold

        return y_pred_classes
    
    def predict_proba(self, X):
        if self.algorithm_ == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            proba = sum(self._samme_proba(estimator, self.n_classes_, X)
                        for estimator in self.estimators_)
        else:  # self.algorithm == "SAMME"
            proba = sum(estimator.predict_proba(X) * w
                        for estimator, w in zip(self.estimators_,
                                                self.estimator_weights_))

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba

def create_ann(input_dim):
    classifier = Sequential([
        Dense(units=64, input_dim=input_dim, kernel_initializer='uniform', activation='relu'),
        Dense(units=32, kernel_initializer='uniform', activation='relu'),
        Dense(units=32, kernel_initializer='uniform', activation='relu'),
        Dense(units=16, kernel_initializer='uniform', activation='relu'),
        Dense(units=8, kernel_initializer='uniform', activation='relu'),
        Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
    ])
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


def run_ann(file_path, target='Class'):
    # Load and preprocess data
    data = pd.read_csv(file_path)
    X = data.drop(target, axis=1)
    y = data[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Handle class imbalance
    smote_enn = SMOTEENN(random_state=42)
    X_train_smote_enn, y_train_smote_enn = smote_enn.fit_resample(X_train, y_train)

    # Now that X_train_smote_enn is defined, get its shape for input_dim
    input_dim = X_train_smote_enn.shape[1]  # This should work now
    base_nn_model = create_ann(input_dim)

    # Continue with AdaBoostClassifier logic...
    adaboost_classifier = AdaBoostClassifier(base_estimator=base_nn_model, n_estimators=30, learning_rate=1.0)
    # Note: You'll need to adjust AdaBoostClassifier to work with Keras models if you haven't already
    adaboost_classifier.fit(X_train_smote_enn, y_train_smote_enn, batch_size=100)  # Example batch_size, adjust as needed

    y_pred = adaboost_classifier.predict(X_test) # no estimators trained error
    y_pred_classes = (y_pred > 0.5).astype(int)  # Converting probabilities to class labels

    # Generating confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=['Non-Fraudulent', 'Fraudulent']))

    # Calculating recall for fraudulent transactions
    print(f"Recall for Fraudulent Transactions: {recall_score(y_test, y_pred_classes):.2f}")

    # Calculating and printing the AUC
    auc = roc_auc_score(y_test, y_pred.ravel())  # Here y_pred is used directly to calculate AUC
    print(f"AUC: {auc:.2f}")

    # Evaluate predictions
    # Implement evaluation using accuracy, precision, recall, F1 score, confusion matrix, etc.


