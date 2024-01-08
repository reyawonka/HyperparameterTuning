import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
import pickle

data_df = pd.read_csv("reduceddataset.csv")
data_df['target'] = np.where(data_df['label'] == 'malware', 1, 0)
features_df = data_df.drop(['MD5', 'label', 'target'], axis=1)

k_best_selector = SelectKBest(score_func=chi2, k=15)
selected_features = k_best_selector.fit_transform(features_df, data_df['target'])

classifier = SVC(kernel='rbf')
classifier.fit(selected_features, data_df['target'])

with open("saved_detector.pkl", "wb") as model_file:
    pickle.dump((classifier, selected_features), model_file)

