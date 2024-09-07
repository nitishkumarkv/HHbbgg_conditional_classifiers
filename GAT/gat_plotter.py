from gat import GAT

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm
import copy
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import sklearn.model_selection

def load_gat_checkpoint(file_path, model, optimizer, scheduler):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_loss_hist = checkpoint['train_loss_hist']
    val_loss_hist = checkpoint['val_loss_hist']
    train_acc_hist = checkpoint['train_acc_hist']
    val_acc_hist = checkpoint['val_acc_hist']
    best_weights = checkpoint['best_weights']
    best_loss = checkpoint['best_loss']
    start_epoch = checkpoint['epoch']
    counter = checkpoint['counter']
    print(f'Checkpoint loaded from {file_path}, resuming from epoch {start_epoch + 1}')
    return start_epoch, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, best_weights, best_loss, counter

input_path = '/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/GAT/inputs'

X_train = torch.load(f'{input_path}/X_train')
X_val = torch.load(f'{input_path}/X_val')
X_test = torch.load(f'{input_path}/X_test')
y_train = torch.load(f'{input_path}/y_train')
y_val = torch.load(f'{input_path}/y_val')
y_test = torch.load(f'{input_path}/y_test')
class_weights_for_training = torch.load(f'{input_path}/class_weights_for_training')
class_weights_for_val = torch.load(f'{input_path}/class_weights_for_val')
class_weights_for_test = torch.load(f'{input_path}/class_weights_for_test')

def generate_edges(num_nodes):
    # create a fully connected graph
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j: 
                edges.append((i, j))
    return torch.tensor(edges).t()

train_list=[]
for features, label, weight in zip(X_train, y_train, class_weights_for_training):
    y = torch.tensor(label, dtype=torch.long) 
    num_nodes = features.shape[0]
    edge_index = generate_edges(num_nodes)
    weight_tensor = torch.tensor(weight, dtype=torch.float)
    data = Data(x=features, edge_index=edge_index, y=y, weight=weight_tensor)
    train_list.append(data)

val_list=[]
for features, label, weight in zip(X_val, y_val, class_weights_for_val):
    y = torch.tensor(label, dtype=torch.long) 
    num_nodes = features.shape[0]
    edge_index = generate_edges(num_nodes)
    weight_tensor = torch.tensor(weight, dtype=torch.float)
    data = Data(x=features, edge_index=edge_index, y=y, weight=weight_tensor)
    val_list.append(data)

batch_size = 512

train_loader = DataLoader(train_list, batch_size, shuffle=True)
val_loader = DataLoader(val_list, batch_size, shuffle=False)

batches_per_epoch = len(train_loader)

classes=["non resonant bkg", "ttH bkg", "GluGluToHH sig", "VBFToHH sig"]

in_dim = 10
hidden_dim = 64
out_dim = 4
num_classes = 4
num_heads = 2

n_epochs = 4

#best_loss = np.inf
#best_weights = None
patience = 50
#counter = 0

model = GAT(hidden_dim=hidden_dim , out_dim=out_dim, num_heads=num_heads)
loss_fn = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr = 8.41439611133144e-05, weight_decay=1.601106283960543e-05)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)

path_to_checkpoint = "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/GAT/gat_checkpoint/"
start_epoch, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, best_weights, best_loss, counter = load_gat_checkpoint(f'{path_to_checkpoint}/gat.pth', model, optimizer, scheduler)

# train_loss_hist = []
# train_acc_hist = []
# val_loss_hist = []
# val_acc_hist = []

for epoch in range(start_epoch + 1, n_epochs):
    batch_loss = []
    batch_acc = []
    model.train()
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for batch_idx, batch in enumerate(train_loader):
            x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch
            out = model(x, edge_index, batch_index)
            y_flat = batch.y
            num_rows = y_flat.size(0) // num_classes
            y_one_hot = y_flat.view(num_rows, num_classes)
            y = torch.argmax(y_one_hot, dim=1)
            weights_batch = batch.weight 
            optimizer.zero_grad()
            loss = loss_fn(out, y)
            weighted_loss = (loss * weights_batch).sum() / weights_batch.sum()
            weighted_loss.backward()
            optimizer.step()
            acc = (torch.argmax(out, 1) == y).float().mean()
            batch_loss.append(float(weighted_loss))
            batch_acc.append(float(acc))
            bar.set_postfix(
                loss=float(weighted_loss),
                acc=float(acc)
            )
    
    train_loss_hist.append(np.mean(batch_loss))
    train_acc_hist.append(np.mean(batch_acc))

    model.eval()
    val_batch_loss = []
    val_batch_acc = []
    with torch.no_grad():
        for batch in val_loader:
            x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch
            out = model(x, edge_index, batch_index)
            y_flat = batch.y
            num_rows = y_flat.size(0) // num_classes
            y_one_hot = y_flat.view(num_rows, num_classes)
            y = torch.argmax(y_one_hot, dim=1)
            weights_batch = batch.weight
            val_loss = loss_fn(out, y)
            weighted_val_loss = (val_loss * weights_batch).sum() / weights_batch.sum()
            val_acc = (torch.argmax(out, dim=1) == y).float().mean()
            val_batch_loss.append(float(weighted_val_loss))
            val_batch_acc.append(float(val_acc))

    ce = np.mean(val_batch_loss)
    acc = np.mean(val_batch_acc)
    val_loss_hist.append(float(ce))
    val_acc_hist.append(float(acc))

    #save the best parameters
    if ce < best_loss:
        best_loss = ce
        best_weights = copy.deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

    print(f"Epoch {epoch} validation: Cross-entropy={float(ce)}, Accuracy={float(acc)}")

path_for_plots = "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/GAT/performance_test"

#plot loss function
plt.plot(train_loss_hist, label="train")
plt.plot(val_loss_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.savefig(f'{path_for_plots}/loss_plot')
plt.clf()

#plot accuracy
plt.plot(train_acc_hist, label="train")
plt.plot(val_acc_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig(f'{path_for_plots}/acc_plot')
plt.clf()