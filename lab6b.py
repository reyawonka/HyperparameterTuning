import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, recall_score

def read(file_path):
    data = pd.read_csv(file_path)
    return data

def datafresh(data):
    label_mapping = {'malware': 1, 'benignware': 0}
    data['label'] = data['label'].map(label_mapping)
    featss = data.drop(["MD5", "Target", "label"], axis=1)
    labels = data["label"]
    return featss, labels

def selfeats(featss, labels, k=15):
    selector = SelectKBest(score_func=chi2, k=k)
    selected_featss = selector.fit_transform(featss, labels)
    return selected_featss

def main():
    data = read("reduceddataset.csv")
    featss, labels = datafresh(data)
    selected_featss = selfeats(featss, labels, k=15)

    model = SVC()
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    space = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [100, 10, 1.0, 0.1, 0.001],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    }

    search = RandomizedSearchCV(model, space, n_iter=500, scoring=make_scorer(recall_score), cv=kfold, random_state=42, n_jobs=-1)

    search.fit(selected_featss, labels)

    print("Best Parameters - ", search.best_params_)
    print("Best Score - ", search.best_score_)

if __name__ == "__main__":
    main()

