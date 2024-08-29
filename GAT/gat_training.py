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

input_path = '/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/GAT/training_inputs_for_gat'

X_train = np.load(f'{input_path}/X_train.npy')
X_val = np.load(f'{input_path}/X_val.npy')
X_test = np.load(f'{input_path}/X_test.npy')
y_train = np.load(f'{input_path}/y_train.npy')
y_val = np.load(f'{input_path}/y_val.npy')
y_test = np.load(f'{input_path}/y_test.npy')
rel_w_train = np.load(f'{input_path}/rel_w_train.npy')
rel_w_val = np.load(f'{input_path}/rel_w_val.npy')
rel_w_test = np.load(f'{input_path}/rel_w_test.npy')

count_not_six_rows = sum(tensor.shape[0] != 6 for tensor in X_test)
print(count_not_six_rows)

print(type(rel_w_train))

print(X_train[-2], rel_w_train)

classes=["non resonant bkg", "ttH bkg", "GluGluToHH sig", "VBFToHH sig"]

def generate_edges(num_nodes):
    # create a fully connected graph
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j: 
                edges.append((i, j))
    return torch.tensor(edges).t()

train_list=[]
for features, label in zip(X_train, y_train):
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(label, dtype=torch.long) 
    num_nodes = x.shape[0]
    edge_index = generate_edges(num_nodes)
    data = Data(x=x, edge_index=edge_index, y=y)
    train_list.append(data)

val_list=[]
for features, label in zip(X_val, y_val):
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(label, dtype=torch.long) 
    num_nodes = x.shape[0]
    edge_index = generate_edges(num_nodes)
    data = Data(x=x, edge_index=edge_index, y=y)
    val_list.append(data)

batch_size = 256

train_loader = DataLoader(train_list, batch_size, shuffle=True)
val_loader = DataLoader(val_list, batch_size, shuffle=False)

in_dim = 8
hidden_dim = 64
out_dim = 4
num_classes = 4
num_heads = 2

n_epochs = 300
batches_per_epoch = len(train_loader)

best_loss = np.inf
best_weights = None
patience = 50
counter = 0

model = GAT(in_dim, hidden_dim, out_dim, num_heads)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 8.41439611133144e-05, weight_decay=1.601106283960543e-05)
best_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)

train_loss_hist = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []

for epoch in range(n_epochs):
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
            optimizer.zero_grad()
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            acc = (torch.argmax(out, 1) == y).float().mean()
            batch_loss.append(float(loss))
            batch_acc.append(float(acc))
            bar.set_postfix(
                loss=float(loss),
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
            val_loss = loss_fn(out, y)
            val_acc = (torch.argmax(out, dim=1) == y).float().mean()
            val_batch_loss.append(float(val_loss))
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

path_for_plots = "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/GAT/plots"

#plot loss function
plt.plot(train_loss_hist, label="train")
plt.plot(val_loss_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.savefig(f'{path_for_plots}/loss_plot_test')
plt.clf()

#plot accuracy
plt.plot(train_acc_hist, label="train")
plt.plot(val_acc_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig(f'{path_for_plots}/acc_plot_test')
plt.clf()

# ROC and CM
model.load_state_dict(best_weights)
model.eval()
y_pred_list = []
y_val_list = []
with torch.no_grad():
    for batch in val_loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        y_pred_list.append(out)
        y_flat = batch.y
        num_rows = y_flat.size(0) // num_classes
        y_one_hot = y_flat.view(num_rows, num_classes)
        y = torch.argmax(y_one_hot, dim=1)
        y_val_list.append(y)

y_pred_val = torch.cat(y_pred_list, dim=0)
y_val_np = torch.cat(y_val_list, dim=0)
y_pred_np = torch.argmax(y_pred_val, 1).cpu().numpy()


threshhold=0.5
mask = torch.max(y_pred_val, dim=1)[0] > threshhold
y_pred_flt = y_pred_np[mask.cpu().numpy()]
y_val_flt = y_val_np[mask.cpu().numpy()]

# Plot confusion matrix
cm = confusion_matrix(y_val_flt, y_pred_flt, normalize='true')
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='.2g', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix, threshhold = {threshhold}')
plt.savefig(f'{path_for_plots}/cm_plot')
plt.clf()

#ROC one vs. all
y_val_bin = label_binarize(y_val_np, classes = range(num_classes))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_pred_val[:, i].cpu().detach())
    # Ensure fpr is strictly increasing
    fpr[i], tpr[i] = zip(*sorted(zip(fpr[i], tpr[i])))
    fpr[i] = np.array(fpr[i])
    tpr[i] = np.array(tpr[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure()
colors = ['royalblue', 'darkorange', 'darkviolet', 'seagreen']
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})' ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (One-vs-All)')
plt.legend(loc="lower right")
plt.savefig(f'{path_for_plots}/roc_plot')
plt.clf()

# #ROC one vs. one
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# combinations_to_plot = [(2, 0), (2, 1), (3, 0), (3, 1)]
# for (i, j) in combinations_to_plot:
#     # Extract the binary labels for classes i and j
#     mask = np.logical_or(y_val_np == i, y_val_np == j)
#     y_true_bin = y_val_bin[mask][:, [i, j]]
#     y_scores = y_pred_val[mask].cpu().detach().numpy()

#     # True labels: i -> 0, j -> 1
#     y_true = np.argmax(y_true_bin, axis=1)
#     y_score = y_scores[:, j]  # Score for class j

#     # Compute ROC curve and ROC area for this pair
#     fpr[(i, j)], tpr[(i, j)], _ = roc_curve(y_true, y_score)
#     roc_auc[(i, j)] = auc(fpr[(i, j)], tpr[(i, j)])
    
#     # Plot the ROC curve
#     plt.figure()
#     plt.plot(fpr[(i, j)], tpr[(i, j)], color='royalblue', lw=2,
#              label=f'ROC curve (area = {roc_auc[(i, j)]:.2f})')
#     plt.plot([0, 1], [0, 1], 'k--', lw=2)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC for class {classes[i]} vs {classes[j]}')
#     plt.legend(loc="lower right")
#     plt.savefig(f'{path_for_plots}/roc_ovo_{i}{j}plot')
#     plt.clf()