# https://www.sciencedirect.com/science/article/pii/S2666285X21000066#bib0013

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)  # Adjusted based on your dataset/input size
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

def prepare_data(data, selected_features, target):
    X = data[selected_features].values if selected_features else data.drop(target, axis=1).values
    y = data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_fix = X_train.astype(float)
    X_test_fix = X_test.astype(float)
    y_train_reshaped = y_train.reshape(-1, 1).astype(int)
    y_test_reshaped = y_test.reshape(-1, 1).astype(int)

    print(f"X_train shape: {X_train.shape}\nX_test shape: {X_test.shape}\nytrain shape: {y_train_reshaped.shape}\nytest shape: {y_test_reshaped.shape}\n")
    X_train_tensor = torch.FloatTensor(X_train_fix)
    print('1\n')
    X_test_tensor = torch.FloatTensor(X_test_fix)
    print('2\n')
    y_train_tensor = torch.LongTensor(y_train_reshaped)
    print('3\n')
    y_test_tensor = torch.LongTensor(y_test_reshaped)
    print('4\n')
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def run_neural_network(data, selected_features, target):
    print("before preparing data\n")
    X_train, X_test, y_train, y_test = prepare_data(data, selected_features, target)
    input_size = X_train.shape[1]
    print("about to initialize model\n")
    model = SimpleNN(input_size)
    print("model initialized\n")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('about to load data\n')
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    print("Made it!\n")
    # Training
    for epoch in range(10):  # Assuming 10 epochs; adjust as needed
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    # Evaluation
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        total = y_test.size(0)
        correct = (predicted == y_test).sum().item()
        print(f'Accuracy: {100 * correct / total}%')

def run_neural_network_with_all_features(file_path, target):
    data = pd.read_csv(file_path)
    run_neural_network(data, None, target)  # Pass None to use all features

def run_neural_network_with_selected_features(file_path_normalized, selected_features, target):
    data = pd.read_csv(file_path_normalized)
    run_neural_network(data, selected_features, target)  # Pass selected features list
