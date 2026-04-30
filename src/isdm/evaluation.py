import numpy as np
import torch
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def predict_logits(model, X, device, batch_size: int = 1024):
    model.eval()
    outputs = []

    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        logits = model(xb)
        outputs.append(logits.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def multilabel_lists_to_dense(y_lists, num_classes: int):
    '''
    Convert a list of lists of label indices into a dense binary matrix.
    '''
    y = np.zeros((len(y_lists), num_classes), dtype=np.uint8)
    for i, labels in enumerate(y_lists):
        if len(labels) > 0:
            y[i, np.asarray(labels, dtype=np.int64)] = 1
    return y


def per_species_auc_sparse(logits, y_lists, num_classes: int):
    y_true = multilabel_lists_to_dense(y_lists, num_classes)
    aucs = {}

    for j in range(num_classes):
        col = y_true[:, j]
        if col.min() == col.max():
            aucs[j] = np.nan
        else:
            aucs[j] = roc_auc_score(col, logits[:, j])

    return aucs