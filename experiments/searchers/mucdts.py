import time
from typing import Tuple, Dict, Any
from mucdts import search as mucdts_search
from searchers.binary_classification_tree import BinaryClassificationTree


def run(
        X_train,
        y_train,
        c: float,
        k: float,
        regularization: float,
        T: int,
    ) -> Tuple[BinaryClassificationTree, float]:
    assert(((X_train == 0) | (X_train == 1)).all())
    assert(((y_train == 0) | (y_train == 1)).all())

    start = time.perf_counter()
    sol = mucdts_search(X_train, y_train, c, T, regularization, k)
    end = time.perf_counter()

    tree = parse(sol.tree)
    tree.fit(X_train, y_train)

    return tree, end - start


def parse(tree: str) -> BinaryClassificationTree:
    return BinaryClassificationTree.parse(tree)
