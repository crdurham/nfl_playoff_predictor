import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import \
(LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

from collections import Counter

def fit_weighted_log(X, y):
    X_sm = sm.add_constant(X)

    class_counts = Counter(y)
    n_total = len(y)

    sample_weights = y.map({
    0: n_total / (2 * class_counts[0]),
    1: n_total / (2 * class_counts[1])
    })

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        freq_weights=sample_weights
    )

    results = model.fit()
    return results

def rfe_features(X,y, n_features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(solver='liblinear')

    rfe = RFE(estimator=lr, n_features_to_select=n_features)
    rfe.fit(X_scaled, y)

    selected_features = X.columns[rfe.support_]
    return selected_features

def lasso_coefficients(X,y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LogisticRegressionCV(
        Cs=10,
        penalty='l1',
        solver='liblinear',
        cv=5,
        scoring='accuracy',
        random_state=42
    )

    lasso.fit(X_scaled, y)
    coeff = pd.Series(lasso.coef_[0], index=X.columns)
    return coeff


