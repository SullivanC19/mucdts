import time
from typing import Dict, Any, Tuple
from pydl85 import DL85Classifier, Cache_Type
import numpy as np

from searchers.binary_classification_tree import BinaryClassificationTree

sparsity_penalty = 0.0
P = N = 0

def sparse_bacc_error(sup_iter):
    supports = list(sup_iter)
    vals = np.array([supports[0] / N, supports[1] / P])
    pred = np.argmax(vals)
    return 0.5 * vals[pred ^ 1] + sparsity_penalty, pred

def run(
        X_train,
        y_train,
        regularization: float = 0.0,
        time_limit: int = 0,
        depth: int = -1,
    ) -> Tuple[BinaryClassificationTree, float, bool]:
    assert(((X_train == 0) | (X_train == 1)).all())
    assert(((y_train == 0) | (y_train == 1)).all())
    
    global P, N, sparsity_penalty
    P = np.sum(y_train == 1)
    N = np.sum(y_train == 0)
    sparsity_penalty = regularization

    start = time.perf_counter()
    clf = DL85Classifier(
      fast_error_function=sparse_bacc_error,
      max_depth=depth,
      time_limit=time_limit,
    )
    clf.fit(X_train, y_train)
    end = time.perf_counter()

    tree = parse(clf)
    tree.fit(X_train, y_train)

    return tree, end - start, clf.timeout_


def parse(clf: DL85Classifier) -> BinaryClassificationTree:
    def parse_node(node: dict) -> BinaryClassificationTree:
        if "value" in node:
            return BinaryClassificationTree()
        return BinaryClassificationTree(
                parse_node(node["right"]),
                parse_node(node["left"]),
                int(node["feat"]))
    return parse_node(clf.base_tree_)
