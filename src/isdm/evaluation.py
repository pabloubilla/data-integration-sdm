import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from pathlib import Path


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

def per_site_auc_sparse(logits, y_lists, num_classes: int):
    ### Important to keep into account for per_site 
    ### Number of classes can be much bigger than the ones actually present on Y Test, as we are also modelling species that might be only present on PO.
    ### This could have an effect on the AUC
    
    y_true = multilabel_lists_to_dense(y_lists, num_classes)
    aucs = {}

    for i in range(len(y_lists)):
        row = y_true[i]
        if row.min() == row.max():
            aucs[i] = np.nan
        else:
            aucs[i] = roc_auc_score(row, logits[i])

    return aucs


class LogitsStore:
    """
    Collects logits and ground-truth labels during a sweep,
    then saves them as a single .npz per model type.

    Structure of saved .npz:
        y_lists/       same shape, each entry is a list of sparse label arrays
        split_ids/     string array [n_splits]
        distances/     float array [n_splits]
        test_numbers/  int array [n_splits]
        options/       string array [n_splits]
    """

    def __init__(self):
        self._records: list[dict] = []

    def add(
        self,
        *,
        split_id: str,
        option: str,
        distance: float,
        test_number: int,
        logits: np.ndarray,       # [n_test, n_classes]
        y_lists: list,            # sparse labels for that test set
    ):
        self._records.append(dict(
            split_id=split_id,
            option=option,
            distance=distance,
            test_number=test_number,
            logits=logits,
            y_lists=y_lists,
        ))

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        n = len(self._records)
        split_ids   = np.array([r["split_id"]    for r in self._records])
        options     = np.array([r["option"]       for r in self._records])
        distances   = np.array([r["distance"]     for r in self._records], dtype=np.float64)
        test_numbers = np.array([r["test_number"] for r in self._records], dtype=np.int64)

        # Ragged arrays: store as object arrays
        logits_arr  = np.empty(n, dtype=object)
        y_lists_arr = np.empty(n, dtype=object)
        for i, r in enumerate(self._records):
            logits_arr[i]  = r["logits"]
            y_lists_arr[i] = r["y_lists"]

        np.savez(
            path,
            split_ids=split_ids,
            options=options,
            distances=distances,
            test_numbers=test_numbers,
            logits=logits_arr,
            y_lists=y_lists_arr,
        )
        print(f"Saved {n} splits to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "LogitsStore":
        data = np.load(path, allow_pickle=True)
        store = cls()
        for i in range(len(data["split_ids"])):
            store.add(
                split_id=str(data["split_ids"][i]),
                option=str(data["options"][i]),
                distance=float(data["distances"][i]),
                test_number=int(data["test_numbers"][i]),
                logits=data["logits"][i],
                y_lists=data["y_lists"][i],
            )
        return store