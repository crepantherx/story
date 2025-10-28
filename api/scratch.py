import numpy as np
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from collections import Counter
from dataclasses import dataclass
import numpy as np
from typing import List


def gini_impurity(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return 1 - np.sum(probabilities ** 2)

def gini_index(left_labels, right_labels):
    total = len(left_labels) + len(right_labels)
    left_weight = len(left_labels) / total
    right_weight = len(right_labels) / total

    return (
        left_weight * gini_impurity(left_labels)
        + right_weight * gini_impurity(right_labels)
    )

@dataclass
class DecisionTreeNode:
    feature_index: int = None
    threshold: int = None
    value: Optional[str] = None

    left_child: Optional["DecisionTreeNode"] = None
    right_child: Optional["DecisionTreeNode"] = None

    def is_leaf_node(self):
        return self.value is not None

@dataclass
class DecisionTreeClassifier:
    depth: Optional[int] = 0
    max_depth: Optional[int] = 5
    min_samples_required_for_split: int = 2

    model = None

    def fit(self, X, Y):
        self.model = self._build_tree(X, Y, self.depth, self.max_depth, self.min_samples_required_for_split)
        return self.model

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return [self._traverse_tree(self.model, x) for x in X]

    def _find_optimal_feature_index_and_threshold_to_split(self, X, Y):
        best_gini_score = float('inf')
        best_feature = None
        best_threshold = None

        _, features = X.shape

        for feature_index in range(features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X1, Y1, X2, Y2 = self._split_based_feature_and_threshold(X, Y, feature_index, threshold)

                if len(Y1) == 0 or len(Y2) == 0:
                    continue

                current_gini_score = gini_index(Y1, Y2)

                if current_gini_score < best_gini_score:
                    best_gini_score = current_gini_score
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _split_based_feature_and_threshold(self, X, Y, feature_index, threshold):
        cond = X[:, feature_index] <= threshold
        return X[cond], Y[cond], X[~cond], Y[~cond]

    def _majority_vote(self, labels):
        return Counter(labels).most_common(1)[0][0]

    def _build_tree(self, X, Y, depth, max_depth, min_samples_required_for_split):

        samples_in_dataset = len(Y)
        classes_in_dataset = len(np.unique(Y))

        if (
            classes_in_dataset == 1
            or depth >= max_depth
            or samples_in_dataset < min_samples_required_for_split
        ):
            return DecisionTreeNode(feature_index=None, threshold=None, value=self._majority_vote(Y), left_child=None,
                                    right_child=None)

        feature_index, threshold = self._find_optimal_feature_index_and_threshold_to_split(X, Y)
        if feature_index is None:
            return DecisionTreeNode(feature_index=None, threshold=None, value=self._majority_vote(Y), left_child=None, right_child=None)

        X1, Y1, X2, Y2 = self._split_based_feature_and_threshold(X, Y, feature_index, threshold)

        left_child = self._build_tree(X1, Y1, depth + 1, max_depth, min_samples_required_for_split)
        right_child = self._build_tree(X2, Y2, depth + 1, max_depth, min_samples_required_for_split)

        return DecisionTreeNode(feature_index, threshold, None, left_child, right_child)

    def _traverse_tree(self, node, x):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] < node.threshold:
            return self._traverse_tree(node.left_child, x)
        else:
            return self._traverse_tree(node.right_child, x)

@dataclass
class RandomForestClassifier:
    n_estimators: int = 10
    max_depth: int = 5
    min_samples_required_for_split: int = 2
    max_features: str = "log2"
    trees: List[DecisionTreeClassifier] = None

    def _bootstrap_sample(self, X, y):
        samples, _ = X.shape
        indices = np.random.choice(samples, size=samples, replace=True)
        return X[indices], y[indices]

    def _get_feature_subset(self, X):
        _, features = X.shape
        if self.max_features == "sqrt":
            return int(np.sqrt(features))
        elif self.max_features == "log2":
            return int(np.log2(features))
        else:
            return features

    def fit(self, X, Y):
        self.trees = []
        for _ in range(self.n_estimators):

            X_sample, Y_sample = self._bootstrap_sample(X, Y)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_required_for_split=self.min_samples_required_for_split
            )

            tree.fit(X_sample, Y_sample)

            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)

@dataclass
class XGBoostTreeNode:
    feature_index: int = None
    threshold: float = None
    value: Optional[float] = None
    left_child: Optional["XGBoostTreeNode"] = None
    right_child: Optional["XGBoostTreeNode"] = None

    def is_leaf(self):
        return self.value is not None

class XGBoostTree:
    def __init__(self, max_depth=3, min_samples_split=2, lambda_=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_ = lambda_
        self.gamma = gamma
        self.root = None

    def _gain(self, G_L, H_L, G_R, H_R):
        def calc(G, H):
            return (G ** 2) / (H + self.lambda_)
        total_gain = calc(G_L, H_L) + calc(G_R, H_R) - calc(G_L + G_R, H_L + H_R)
        return 0.5 * total_gain - self.gamma

    def _split(self, X, g, h):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_mask = None

        n_samples, n_features = X.shape
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                mask = X[:, feature] <= t
                if np.sum(mask) < self.min_samples_split or np.sum(~mask) < self.min_samples_split:
                    continue
                G_L, H_L = np.sum(g[mask]), np.sum(h[mask])
                G_R, H_R = np.sum(g[~mask]), np.sum(h[~mask])
                gain = self._gain(G_L, H_L, G_R, H_R)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t
                    best_left_mask = mask
        return best_feature, best_threshold, best_left_mask

    def _build(self, X, g, h, depth):
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            leaf_value = -np.sum(g) / (np.sum(h) + self.lambda_)
            return XGBoostTreeNode(value=leaf_value)

        feature, threshold, mask = self._split(X, g, h)
        if feature is None:
            leaf_value = -np.sum(g) / (np.sum(h) + self.lambda_)
            return XGBoostTreeNode(value=leaf_value)

        left = self._build(X[mask], g[mask], h[mask], depth + 1)
        right = self._build(X[~mask], g[~mask], h[~mask], depth + 1)
        return XGBoostTreeNode(feature_index=feature, threshold=threshold, left_child=left, right_child=right)

    def fit(self, X, g, h):
        self.root = self._build(X, g, h, depth=0)

    def _predict_row(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_row(x, node.left_child)
        else:
            return self._predict_row(x, node.right_child)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_row(x, self.root) for x in X])

@dataclass
class XGBoostClassifier:
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    lambda_: float = 1.0
    gamma: float = 0.0
    min_samples_split: int = 2

    trees: List[List[XGBoostTree]] = None  # list of boosting rounds, each round has K trees
    init_score: float = 0.0
    n_classes: int = 2

    def __post_init__(self):
        self.trees = []

    def _softmax(self, F):
        exp_F = np.exp(F - np.max(F, axis=1, keepdims=True))
        return exp_F / exp_F.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.n_classes = len(np.unique(y))

        # Initialize F_m with log-odds per class (or 0 for multiclass)
        F_m = np.zeros((X.shape[0], self.n_classes))

        for m in range(self.n_estimators):
            round_trees = []
            # Compute softmax probabilities
            p = self._softmax(F_m)

            for k in range(self.n_classes):
                g = p[:, k] - (y == k).astype(float)   # gradient
                h = p[:, k] * (1 - p[:, k])           # hessian

                tree = XGBoostTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    lambda_=self.lambda_,
                    gamma=self.gamma
                )
                tree.fit(X, g, h)
                update = tree.predict(X)
                F_m[:, k] += self.learning_rate * update
                round_trees.append(tree)
            self.trees.append(round_trees)

    def predict_proba(self, X):
        X = np.array(X)
        F = np.zeros((X.shape[0], self.n_classes))
        for round_trees in self.trees:
            for k, tree in enumerate(round_trees):
                F[:, k] += self.learning_rate * tree.predict(X)
        return self._softmax(F)

    def predict(self, X):
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)
