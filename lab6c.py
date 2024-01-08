import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV


data = pd.read_csv("reduceddataset.csv")
featss = data.drop(["MD5", "label", "Target"], axis=1)
labels = data["label"]


grid = {
    'kernel': ["linear", "poly", "rbf", "sigmoid"],
    'C': [100, 10, 1.0, 0.1, 0.001],
    'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10, 100]
}


model = SVC()


foldd = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


searchgrid = GridSearchCV(model, grid, scoring='accuracy', cv=foldd)
searchgrid.fit(featss, labels)

print("Best Score:", searchgrid.best_score_)
print("Best Parameters:", searchgrid.best_params_)

