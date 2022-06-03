import numpy as np
import pandas as pd


def value_split_binary(x, y, w, fid, root):
    root = root
    left_child = [i for i in root if x.iloc[i][fid] == 0]
    right_child = [i for i in root if x.iloc[i][fid] == 1]
    return information_gain(y, w, root, left_child, right_child), .5


def entropy(y, w, ids):
    """
    Returns the entropy in the labels for the data points in ids.

    :@param y: all labels
    :@param ids: the indexes of data points
    """
    if len(ids) == 0:  # deal with corner case when there is no data point.
        return 0
    pos_ids = y.iloc[ids] > 0
    pos_sum = np.sum(pos_ids * w[ids])
    total = np.sum(w[ids])
    p = [pos_sum/total, 1 - (pos_sum/total)]
    if any(np.isclose(p, 0)):
        return 0
    return -np.sum(p * np.log2(p))


def information_gain(y, w, root, left_child, right_child):
    """
    Returns the information gain by splitting root into left child and right child.

    :@param y: all labels
    :@param root: indexes of all the data points in the root
    :@param left_child: the subset of indexes in the left child
    :@param right_child: the subset of indexes in the right child
    """
    total = float(len(root))
    return entropy(y, w, root) \
           - len(left_child) / total * entropy(y, w, left_child) \
           - len(right_child) / total * entropy(y, w, right_child)


def value_split_continuous(x, y, w, fid, root):
    """
    Return the best value and its corresponding threshold by splitting based on a continuous feature.

    :@param x: all feature values
    :@param y: all labels
    :@param fid: feature id to split the tree based on
    :@param root: indexes of all the data points in the root
    :@param criteria_func: the splitting criteria function
    """
    best_value, best_thres = 0, 0
    choices = np.unique(x.iloc[root][fid])
    for thres in choices:
        left_child = x.iloc[root][x.iloc[root][fid] <= thres].index.tolist()
        right_child = x.iloc[root][x.iloc[root][fid] > thres].index.tolist()
        value = information_gain(y, w, root, left_child, right_child)
        if value > best_value:
            best_value, best_thres = value, thres
    return best_value, best_thres


class LeafNode:
    """
    Class for leaf nodes in the decision tree
    """
    def __init__(self, label, count, total):
        """
        :@param label: label of the leaf node
        :@param count: number of data points with class 'label' falling in this leaf
        :@param count: number of datapoints of any label falling in this leaf
        """
        self.label = label
        self.count = count
        self.total = total
        
    def predict(self, x):
        """
        Return predictions for features x

        :@param x: feature values
        """
        return np.full(x.shape[:1], self.label)
    
    def display(self, out_str, depth=0):
        """
        Display contents of a leaf node
        """
        prefix = '\t'*depth
        error = 1.0 - self.count / float(self.total)
        out_str += f'{prefix}leaf: label={self.label}, error={error} ({self.count}/{self.total} correct)\n'
        return out_str

    def features(self):
        return []


class TreeNode:
    """
    Class for internal (non-leaf) nodes in the decision tree
    """
    def __init__(self, feat_id, feat_val):
        """
        :@param feat_id: index of the feature that this node splits on
        :@param feat_val: threshold for the feature that this node splits on
        """
        self.feat_id = feat_id
        self.feat_val = feat_val
        self.left = None
        self.right = None
    
    def split(self, x, root):
        """
        Given the datapoints falling into current node, return two arrays of indices in x corresponding to the
        left and right subtree
        
        :@param x: all feature values
        :@param root: indexes of all the data points in the current node
        """
        root = np.array(root)
        below_thres = x.iloc[root][self.feat_id] <= self.feat_val
        return root[below_thres], root[~below_thres]
    
    def predict(self, x):
        """
        Return an array of predictions for given 'x' for the current node
        
        :@param x: datapoints
        """
        assert self.left is not None and self.right is not None, 'predict called before fit'
        if len(x) == 0:
            return []
        y_pred = np.zeros(x.shape[:1])
        left, right = self.split(x, root=list(range(x.shape[0])))
        y_pred[left], y_pred[right] = self.left.predict(x.iloc[left]), self.right.predict(x.iloc[right])
        return y_pred
    
    def display(self, out_str, depth=0):
        """
        Display contents of a non-leaf node
        """
        prefix = '\t'*depth
        out_str += f'{prefix}{self.feat_id}\n'
        out_str += f'{prefix}x <= {self.feat_val}\n'
        out_str = self.left.display(out_str, depth=depth+1)
        out_str += f'{prefix}x > {self.feat_val}\n'
        out_str = self.right.display(out_str, depth=depth+1)
        return out_str

    def features(self):
        return [self.feat_id] + self.left.features() + self.right.features()


class DecisionTree:
    """
    Class for the decision tree
    """
    def __init__(self, max_depth=1, binary_feat_ids=[]):
        """
        :@param max_depth: Maximum depth that a decision tree can take
        :@param criteria_func: criteria function to split features
        :@param binary_feat_id: list of indexes of binary features
        """
        self.max_depth = max_depth
        self.binary_feat_ids = binary_feat_ids
        self.root = None
        self.x = None
        self.y = None
        self.w = None
        
    def fit(self, x, y, w):
        """
        Fit a tree to the given dataset using a helper function
        """
        self.x = x
        self.y = y
        self.w = w
        self.root = self.fit_helper(list(self.x.index))
    
    def fit_helper(self, root, depth=1):
        """
        Recursive helper function for fitting a decision tree
        Returns a node (can be either LeafNode or TreeNode)
        
        :@param root: array of indices of datapoints which fall into the current node
        :@param depth: current depth of the tree being built
        """
        
        """
        Strategy:
        1. If current partition is pure i.e. labels corresponding to all indices in root are the same
           OR the maximum depth has been reached, stop building the tree and return a LeafNode
        2. If not, find out the best feature to split on along with the threshold, create a TreeNode and
           recursively call fit_helper on the two splits (You can assume the threshold for a binary feature
           to be 0.5). Finally, return the current node
        """
        if np.all(self.y.iloc[root] > 0) or np.all(self.y.iloc[root] < 0) or depth > self.max_depth:
            leaf_sum = np.sum((self.w[root] * self.y.iloc[root]).to_numpy())
            if leaf_sum == 0:
                ys, counts = np.unique(self.y.iloc[root], return_counts=True)
                i_max = np.argmax(counts)
                label = ys[i_max]
            else:
                label = np.sign(leaf_sum)
            count = np.sum(np.sign(self.y.iloc[root]).to_numpy() == label)
            return LeafNode(label, count, len(root))
        res = {}
        for fid in self.x.columns:
            if fid in self.binary_feat_ids:
                res[fid] = value_split_binary(self.x, self.y, self.w,
                                              fid, root)
            else:
                res[fid] = value_split_continuous(self.x, self.y, self.w,
                                                  fid, root)
        # select best feature w.r.t criteria values
        feat_id = max(res, key=res.get)
        feat_val = res[feat_id][1]
        # create node and partition indices for children
        node = TreeNode(feat_id, feat_val)
        left_inds, right_inds = node.split(self.x, root)
        node.left = self.fit_helper(left_inds, depth=depth+1)
        node.right = self.fit_helper(right_inds, depth=depth+1)
        return node
    
    def predict(self, x):
        """
        Return predictions for a given dataset
        """
        assert self.root is not None, 'fit not yet called'
        prediction = self.root.predict(x)
        return prediction
    
    def display(self):
        assert self.root is not None, 'fit not yet called'
        out_str = ""
        out_str = self.root.display(out_str)
        return out_str

    def check(self, X, y):
        y_hat = self.predict(X)
        diff = y_hat != y.to_numpy()
        return diff.astype(int)

    def features(self):
        return self.root.features()
    

class Adaboost:
    def __init__(self, X, y, k, max_depth, binary_feat_ids=[]):
        self.X = X
        self.y = y
        self.k = k
        self.max_depth = max_depth
        self.binary_feat_ids = binary_feat_ids
        self.alphas = []
        self. hs = []

    def train(self):
        weights = np.full(self.y.shape, 1/self.y.shape[0])
        for i in range(self.k):
            h = DecisionTree(self.max_depth, binary_feat_ids=self.binary_feat_ids)
            h.fit(self.X, self.y, weights)
            err = np.sum(weights * h.check(self.X, self.y))/np.sum(weights)
            if err == 0 or err == 1:
                if len(self.alphas) == 0 and len(self.hs) == 0:
                    self.alphas.append(1)
                    self.hs.append(h)
                    print('stopping early: ', i)
                return
            alpha = 1/2*np.log((1 - err)/err)
            self.alphas.append(alpha)
            self.hs.append(h)
            weights *= np.exp(-1*alpha*self.y.to_numpy()*h.predict(self.X))
            weights = weights/np.sum(weights)

    def evaluate(self, x):
        out = 0
        for i in range(len(self.alphas)):
            out += self.alphas[i]*self.hs[i].predict(x)
        return np.sign(out)

    def error(self, x, y):
        y_hat = self.evaluate(x)
        return np.sum(y_hat != y)/len(y)

    def precision(self, x, y):
        true_ids = np.where(y == 1)
        y_hat = self.evaluate(x)
        pred_ids = np.where(y_hat == 1)
        if len(true_ids[0]) == 0 and len(pred_ids[0]) == 0:
            return 1
        elif len(true_ids[0]) == 0 or len(pred_ids[0]) == 0:
            return 0
        return np.sum(y_hat[true_ids] == y.iloc[true_ids])/len(pred_ids[0])

    def recall(self, x, y):
        true_ids = np.where(y == 1)
        y_hat = self.evaluate(x)
        if len(true_ids[0]) == 0:
            return 1
        return np.sum(y_hat[true_ids] == y.iloc[true_ids])/len(true_ids[0])

    def f1(self, x, y):
        p = self.precision(x, y)
        r = self.recall(x, y)
        if p == 0 and r == 0:
            return 0
        return 2*(p*r)/(p + r)

    def display_trees(self):
        for h in self.hs:
            print(h.display())

    def feature_counts(self, names, features):
        for h in self.hs:
            features.loc['n_trees'] += 1
            feat_list = h.features()
            for name in names:
                if name in feat_list:
                    features.loc[name] += 1


def k_fold(fold_size, X, y, k, max_depth, show_trees=False, show_stats=False):
    comb = pd.concat((X, y), axis=1)
    ids = comb.index.to_numpy()
    np.random.shuffle(ids)
    folds = np.array_split(ids, len(comb)//fold_size)
    training_errors = []
    validation_errors = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    col_names = X.columns.to_list() + ['n_trees']
    features = pd.DataFrame(np.zeros(len(col_names)),
                            index=col_names, columns=['Count'])
    for fold in folds:
        x_trn = comb.iloc[X.index.difference(fold)][X.columns]
        x_trn = x_trn.reset_index(drop=True)
        x_trn_no_zip = x_trn.drop(columns='zip')
        y_trn = comb.iloc[y.index.difference(fold)]['Gentrified2020']
        y_trn = y_trn.reset_index(drop=True)
        x_val = comb.iloc[fold][X.columns]
        x_val = x_val.reset_index(drop=True)
        x_val_no_zip = x_val.drop(columns='zip')
        y_val = comb.iloc[fold]['Gentrified2020']
        y_val = y_val.reset_index(drop=True)
        ada = Adaboost(x_trn_no_zip, y_trn, k, max_depth, ['gardens'])
        ada.train()
        ada.feature_counts(col_names, features)
        if show_trees:
            ada.display_trees()
        accuracies.append(1 - ada.error(x_val_no_zip, y_val))
        precisions.append(ada.precision(x_val_no_zip, y_val))
        recalls.append(ada.recall(x_val_no_zip, y_val))
        f1s.append(ada.f1(x_val_no_zip, y_val))
        training_errors.append(ada.error(x_trn_no_zip, y_trn))
        validation_errors.append(ada.error(x_val_no_zip, y_val))
    if show_stats:
        print('Accuracy: ', np.mean(accuracies))
        print('Precision: ', np.mean(precisions))
        print('Recall: ', np.mean(recalls))
        print('F1: ', np.mean(f1s))
    return np.mean(training_errors), np.mean(validation_errors), features


def test_hyperparams(X, y, sizes, num_models, depths):
    training_errs = []
    validation_errs = []
    feat_cts = []
    for size in sizes:
        for num in num_models:
            for depth in depths:
                print('size ', size, ', num ', num, ', depth ', depth)
                trn_err, val_err, feats = k_fold(size, X, y, num, depth)
                training_errs.append([size, num, depth, trn_err])
                validation_errs.append([size, num, depth, val_err])
                feat_cts.append(feats)
    return training_errs, validation_errs, feat_cts
