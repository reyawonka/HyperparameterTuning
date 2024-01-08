# HyperparameterTuning

<!DOCTYPE html>
<html>
<head>
<title>Lab on Hyperparameter Tuning</title>
<style>
    body {font-family: Arial, sans-serif;}
    .code {background-color: #f4f4f4; padding: 10px; border-left: 3px solid #007BFF; margin: 10px 0;}
    h2 {color: #007BFF;}
</style>
</head>
<body>

<h1>Lab on Hyperparameter Tuning</h1>
<p>This repository contains the code and documentation for a lab exercise focused on Hyperparameter Tuning in Machine Learning models, specifically using Support Vector Machines (SVMs).</p>

<h2>Overview of Scripts</h2>

<h3>1. ModelApplication.py</h3>
<p>This script loads a pre-trained classifier and a feature list, then processes files in a specified directory. It's likely used for applying the model to new data.</p>
<pre class="code">
# Sample Code Snippet
with open("saved_detector.pkl", "rb") as f:
    clf, features = pickle.load(f)
#...
</pre>

<h3>2. InitialModelTraining.py</h3>
<p>Handles data processing, feature selection, and initial model training with an SVM. It uses chi-squared feature selection and an rbf kernel for the SVM.</p>
<pre class="code">
# Sample Code Snippet
k_best_selector = SelectKBest(score_func=chi2, k=15)
#...
classifier = SVC(kernel='rbf')
#...
</pre>

<h3>3. CrossValidationSetup.py</h3>
<p>Similar to InitialModelTraining.py, this script is involved in data processing and feature selection but also sets up for cross-validation.</p>
<pre class="code">
# Sample Code Snippet
def selfeats(featss, labels, k=15):
    selector = SelectKBest(chi2, k=k)
    #...
</pre>

<h3>4. RandomizedSearchTuning.py</h3>
<p>Introduces RandomizedSearchCV for hyperparameter tuning of the SVM model, exploring various combinations of parameters.</p>
<pre class="code">
# Sample Code Snippet
RandomizedSearchCV(SVC(), param_distributions=grid, n_iter=100)
#...
</pre>

<h3>5. GridSearchTuning.py</h3>
<p>Uses GridSearchCV for a more exhaustive hyperparameter search over the specified parameter grid for the SVM model.</p>
<pre class="code">
# Sample Code Snippet
grid = {'kernel': ["linear", "poly", "rbf", "sigmoid"], 'C': [100, 10, 1.0, 0.1, 0.001]}
#...
</pre>

<h2>Conclusion</h2>
<p>This lab demonstrates the importance of hyperparameter tuning in machine learning models, showing how different approaches like RandomizedSearchCV and GridSearchCV can significantly impact the performance of an SVM classifier.</p>

</body>
</html>
