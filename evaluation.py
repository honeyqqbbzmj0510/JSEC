from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn import metrics
from Hungarian import Munkres
import numpy as np
from sklearn.metrics import accuracy_score

# 输出聚类指标acc, nmi, adjscore, f1_macro, pur

class linkpred_metrics( ) :
    def __init__(self, edges_pos, edges_neg) :
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg

    def get_roc_score(self, emb, feas) :
        # if emb is None:
        #     feed_dict.update({placeholders['dropout']: 0})
        #     emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x) :
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in self.edges_pos :
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(feas['adj_orig'][e[0], e[1]])

        preds_neg = []
        neg = []
        for e in self.edges_neg :
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(feas['adj_orig'][e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score, emb



def purity_score(y_true, y_pred):


    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


class clustering_metrics( ) :
    def __init__(self, true_label, predict_label) :
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self) :
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)
        # print('gnd:',numclass1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        # print('prd:',numclass2)
        if numclass1 != numclass2 :
            print('Class Not equal, Error!!!!')
            return 0, 0, 0, 0, 0, 0, 0, 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1) :
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2) :
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres( )
        cost = cost.__neg__( ).tolist( )

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1) :
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        PUR = purity_score(self.true_label, new_predict)
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, PUR

    def evaluationClusterModelFromLabel(self) :
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, pur = self.clusteringAcc( )


        return acc, nmi, adjscore, f1_macro, pur