import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.initializers import HeNormal



def run_ann(X_train, X_test, y_train, y_test, kern_init):

    # Defining the model architecture
    if kern_init is 'HeNormal':
        classifier = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(units=64, kernel_initializer=HeNormal(), activation='relu'),
        Dense(units=32, kernel_initializer=HeNormal(), activation='relu'),
        Dense(units=32, kernel_initializer=HeNormal(), activation='relu'),
        Dense(units=16, kernel_initializer=HeNormal(), activation='relu'),
        Dense(units=8, kernel_initializer=HeNormal(), activation='relu'),
        Dense(units=1, kernel_initializer=HeNormal(), activation='sigmoid')
    ])
    else #use kern_init as value 'uniform' or 'normal
        classifier = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(units=64, kernel_initializer=kern_init, activation='relu'),
        Dense(units=32, kernel_initializer=kern_init, activation='relu'),
        Dense(units=32, kernel_initializer=kern_init, activation='relu'),
        Dense(units=16, kernel_initializer=kern_init, activation='relu'),
        Dense(units=8, kernel_initializer=kern_init, activation='relu'),
        Dense(units=1, kernel_initializer=kern_init, activation='sigmoid')
    ])

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the model
    classifier.fit(X_train, y_train, batch_size=64, epochs=5, verbose=1)

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

    # Calculating and printing the AUC
    auc = roc_auc_score(y_test, y_pred.ravel())  # Here y_pred is used directly to calculate AUC
    print(f"AUC: {auc:.2f}")