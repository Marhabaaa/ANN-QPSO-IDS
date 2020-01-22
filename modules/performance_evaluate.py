from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def f_score(p, r):
    return 2 * ((p * r) / (p + r))


def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + fp + tn + fn)


def get_metrics(tp, tn, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    f_s = f_score(p, r)
    a = accuracy(tp, tn, fp, fn)

    return p, r, f_s, a


def get_values(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp, tn, fp, fn


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def get_report(y_true, y_pred):
    return metrics.classification_report(y_true, y_pred)
