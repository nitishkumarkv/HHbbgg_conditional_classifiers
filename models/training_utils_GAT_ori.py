import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.models import GAT
import random
import numpy as np
import os
from GAT import GAT_classifier
import matplotlib.pyplot as plt


def load_data(data_path):
    print(f"INFO: Loading data from {data_path}")
    data_list = torch.load(data_path)
    return data_list

def get_weights_for_training(y_train, rel_w_train):

    class_weights_for_training = ak.zeros_like(rel_w_train)

    for i in range(y_train.shape[1]):

        cls_bool = (y_train[:, i] == 1)
        abs_rel_xsec_weight_for_class = abs(rel_w_train) * cls_bool
        class_weights_for_training = class_weights_for_training + (abs_rel_xsec_weight_for_class / np.sum(abs_rel_xsec_weight_for_class))

    for i in range(y_train.shape[1]):
        print(f"(number of events: sum of class_weights_for_training) for class number {i+1} = ({sum(y_train[:, i])}: {sum(class_weights_for_training[y_train[:, i] == 1])})")

    class_weights_for_training = torch.tensor(class_weights_for_training, dtype=torch.float32).reshape(1, -1)

    return class_weights_for_training

def get_weights_for_val_test(y_val, rel_w_val):

    class_weights_for_val = ak.zeros_like(rel_w_val)

    for i in range(y_val.shape[1]):
        cls_bool = (y_val[:, i] == 1)
        rel_xsec_weight_for_class = rel_w_val * cls_bool
        class_weights_for_val = class_weights_for_val + (rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class))

    for i in range(y_val.shape[1]):
        print(f"(number of events: sum of class_weights_for_val) for class number {i+1} = ({sum(y_val[:, i])}: {sum(class_weights_for_val[y_val[:, i] == 1])})")
    
    class_weights_for_val = torch.tensor(class_weights_for_val, dtype=torch.float32).reshape(1, -1)

    return class_weights_for_val

def shuffle_and_split_data(data_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """shuffle and split the data into train, val, and test sets"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    #assert len(data_list) == len(rel_xsec_weight), "data_list and rel_xsec_weight must be the same length."

    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    #combined = list(zip(data_list, rel_xsec_weight))
    #random.shuffle(combined)
    random.shuffle(data_list)
    
    #data_list_shuffled, rel_xsec_weight_shuffled = zip(*combined)
    
    total_samples = len(data_list)
    train_end = int(train_ratio * total_samples)
    val_end = train_end + int(val_ratio * total_samples)
    
    # Split data_list
    train_data = data_list[:train_end]
    val_data = data_list[train_end:val_end]
    test_data = data_list[val_end:]
    
    # Split rel_xsec_weight
    #train_weights = rel_xsec_weight_shuffled[:train_end]
    #val_weights = rel_xsec_weight_shuffled[train_end:val_end]
    #test_weights = rel_xsec_weight_shuffled[val_end:]

    # modify the weights for training and validation
    #train_weights = get_weights_for_training(train_data.y, train_weights)
    #print()
    #val_weights = get_weights_for_val_test(val_data.y, val_weights)
    #test_weights = get_weights_for_val_test(test_data.y, test_weights)
    
    return train_data, val_data, test_data#, (train_weights, val_weights, test_weights)

def create_data_loaders(train_data, val_data, test_data, batch_size=32, shuffle=True):
    """Create DataLoaders for train, val, and test sets."""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train(model, device, train_loader, optimizer, criterion):
    """Training loop for one epoch."""
    model.train()
    total_loss = 0
    correct = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

        # Compute accuracy
        pred = output.argmax(dim=1)
        #correct += pred.eq(data.y).sum().item()

    average_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    return average_loss, accuracy

def evaluate(model, device, loader, criterion):
    """Evaluation loop for validation or test set."""
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item() * data.num_graphs

            # Assuming classification task
            pred = output.argmax(dim=1)
            #correct += pred.eq(data.y).sum().item()

    accuracy = correct / len(loader.dataset)
    average_loss = total_loss / len(loader.dataset)
    return average_loss, accuracy

def main():
    # Hyperparameters and configurations
    data_path = "/net/scratch_cms3a/kasaraguppe/work/HHbbgg_classifier/HHbbgg_conditional_classifiers/data/gnn_data_lst_full/data_list.pt"
    #data_path = "/net/scratch_cms3a/kasaraguppe/work/HHbbgg_classifier/HHbbgg_conditional_classifiers/data/gnn_data_lst/data_list.pt"
    #rel_xsec_weight_path = torch.load("/net/scratch_cms3a/kasaraguppe/work/HHbbgg_classifier/HHbbgg_conditional_classifiers/data/gnn_data_lst/rel_xsec_weight.pt")
    input_dim = 11
    hidden_dim = 64
    num_init_mlp_layers = 1
    num_final_mlp_layers = 1
    num_gat_layers = 1
    num_message_passing_layers = 1
    num_extra_feat = 0
    output_dim = 4
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    dropout = 0.5
    activation = "leakyrelu"
    seed = 42

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_list = load_data(data_path)
    rel_xsec_weight = data_list #load_data(data_list)

    # Shuffle and split data
    #(train_data, val_data, test_data), (train_weights, val_weights, test_weights) = shuffle_and_split_data(data_list, rel_xsec_weight_path, seed=seed)
    train_data, val_data, test_data = shuffle_and_split_data(data_list, seed=seed)

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data, batch_size)

    # Initialize model
    model = GAT_classifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_init_mlp_layers=num_init_mlp_layers,
        num_final_mlp_layers=num_final_mlp_layers,
        num_gat_layers=num_gat_layers,
        num_message_passing_layers=num_message_passing_layers,
        num_extra_feat=num_extra_feat,
        output_dim=output_dim,
        dropout=dropout,
        activation=activation
    ).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  

    # Lists to store metrics
    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion)
        val_loss, val_accuracy = evaluate(model, device, val_loader, criterion)

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_accuracy = evaluate(model, device, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Plotting training and validation loss
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(7, 5))

    # Plot Loss
    #plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_list, label='Train Loss')
    plt.plot(epochs_range, val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Accuracy
    #plt.subplot(1, 2, 2)
    #plt.plot(epochs_range, train_accuracy_list, label='Train Accuracy')
    #plt.plot(epochs_range, val_accuracy_list, label='Validation Accuracy')
    #plt.xlabel('Epochs')
    #plt.ylabel('Accuracy')
    #plt.title('Training and Validation Accuracy')
    #plt.legend()

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
