from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def precision(y_true, y_pred, average):
    return precision_score(y_true, y_pred, average=average, zero_division=0)

def recall(y_true, y_pred, average):
    return recall_score(y_true, y_pred, average=average, zero_division=0)

def f1(y_true, y_pred, average):
    return f1_score(y_true, y_pred, average=average, zero_division=0)

def confusion_mat(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def auc_roc(y_true, y_pred_proba):
    return roc_auc_score(y_true, y_pred_proba)


def evaluate_classification(y_true, y_pred, y_pred_proba=None, average = "weighted"):
    metrics = {
        'accuracy': accuracy(y_true, y_pred),
        'precision': precision(y_true, y_pred, average=average),
        'recall': recall(y_true, y_pred, average=average),
        'f1_score': f1(y_true, y_pred, average=average),
        'auc_roc': auc_roc(y_true, y_pred_proba) if y_pred_proba else "None",
        'confusion_mat': confusion_mat(y_true, y_pred)
    }
    return metrics

# "binary" | "macro" | "micro" | "samples" | "weighted"