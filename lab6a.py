import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

def readd(file_path):
    return pd.read_csv(file_path)

def datafresh(data):
    
    maplabel = {'malware': 1, 'benignware': 0}
    data['label'] = data['label'].map(maplabel)

   
    featss = data.drop(["MD5", "Target", "label"], axis=1)

    return featss, data["label"]

def selfeats(featss, labels, k=15):
    selector = SelectKBest(score_func=chi2, k=k)
    selected_featss = selector.fit_transform(featss, labels)
    return selected_featss

def grid(firstx, firsty, cv, griddd):
    gridfinals = []
    for kernel in griddd['kernel']:
        for C in griddd['C']:
            svm = SVC(kernel=kernel, C=C)
            scores = cross_val_score(svm, firstx, firsty, cv=cv, scoring='accuracy')
            scoremean = scores.mean()
            std = scores.std()
            gridfinals.append((kernel, C, scoremean, std))
    return gridfinals

def main():
    
    data = readd("reduceddataset.csv")

   
    featss, labels = datafresh(data)

  
    selected_featss = selfeats(featss, labels)

    
    firstx = selected_featss
    firsty = labels

    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    
    griddd = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [100, 10, 1.0, 0.1, 0.001]
    }

    
    gridfinals = grid(firstx, firsty, cv, griddd)

    
    for i, (kernel, C, scoremean, std) in enumerate(gridfinals):
        print(f"Scenario {i + 1}: Kernel- {kernel}, C: {C}")
        print(f"Mean CV Score- {scoremean:.4f}")
        print(f"Standard Deviation- {std:.4f}")

if __name__ == "__main__":
    main()
