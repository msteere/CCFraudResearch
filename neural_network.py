import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE  # Make sure imbalanced-learn is installed

def run_ann(file_path, target='Class', selected_features=None):
    # Loading the data
    data = pd.read_csv(file_path)
    
    # Selecting features based on the provided argument
    if selected_features is not None:
        X = data[selected_features]
    else:
        X = data.drop(target, axis=1)
        
    y = data[target].values

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Applying SMOTE to the training data to handle imbalance
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Defining the model architecture
    classifier = Sequential([
        Input(shape=(X_train_smote.shape[1],)),
        Dense(units=64, kernel_initializer='uniform', activation='relu'),
        Dense(units=32, kernel_initializer='uniform', activation='relu'),
        Dense(units=32, kernel_initializer='uniform', activation='relu'),
        Dense(units=16, kernel_initializer='uniform', activation='relu'),
        Dense(units=8, kernel_initializer='uniform', activation='relu'),
        Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
    ])

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the model
    classifier.fit(X_train_smote, y_train_smote, batch_size=10, epochs=10, verbose=1)

    # Evaluating the model
    scores = classifier.evaluate(X_test, y_test)
    print("\nModel Accuracy: %.2f%%" % (scores[1]*100))

    # Predicting the test set results
    y_pred = classifier.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)  # Converting probabilities to class labels

    # Generating confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=['Non-Fraudulent', 'Fraudulent']))

    # Calculating recall for fraudulent transactions
    print(f"Recall for Fraudulent Transactions: {recall_score(y_test, y_pred_classes):.2f}")
