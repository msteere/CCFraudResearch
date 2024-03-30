import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

def run_ann_with_all_features(file_path, target='Class'):
    data = pd.read_csv(file_path)
    X = data.drop(target, axis=1)
    y = data[target].values

    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Quick sanity check with the shapes of Training and Testing datasets
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # Defining the ANN architecture
    classifier = Sequential()
    classifier.add(Dense(units=16, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the ANN
    classifier.fit(X_train, y_train, batch_size=10, epochs=10, verbose=1)

    # Evaluating the model
    scores = classifier.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))
