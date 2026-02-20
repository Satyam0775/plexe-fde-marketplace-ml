from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

def evaluate_model(model, X_test, y_test):

    prob = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    return {
        "roc_auc": roc_auc_score(y_test, prob),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "f1": f1_score(y_test, pred)
    }