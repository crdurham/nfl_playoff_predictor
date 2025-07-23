import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

import itertools
from collections import Counter

def test_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data



def fit_weighted_log(X, y):
    '''Fit a weighted logistic regression model with feature data X and label data y. Returns the results of
    fitting the model.'''

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
    '''Run recursive feature elimination on feature data X (after scaling) with label data y. Choose number of
    features to select via parameter n_features.'''

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(solver='liblinear')

    rfe = RFE(estimator=lr, n_features_to_select=n_features)
    rfe.fit(X_scaled, y)

    selected_features = X.columns[rfe.support_].tolist()
    return selected_features

def lasso_coefficients(X,y):
    '''Run Lasso logistic regression on feature data X with label data y. Uses/Tries 10 L1 penalty parameter values with 5-fold cross
    validation.'''
    
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

def confusion_matrix_and_classification_rep(fitted_model, X_test, y_test, threshold=0.5):
    #X_test_sm = sm.add_constant(X_test)
    y_prob = fitted_model.predict(X_test)
    y_pred = (y_prob >= threshold).astype(int)

    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    return confusion, report


def scatter_matrix_fn(df, cols):
    sns.pairplot(df[cols], 
                 diag_kind='hist', #Estimate density within diagonal vars
                 diag_kws={'bins': 16, 'color': 'hotpink'},
                 hue=None,
                 height=1.35,
                 aspect=1.35,  
                 plot_kws={'color': 'teal', 'alpha': 0.25, 's': 15})


def labeled_hist(df, cols, yes_color='green', no_color='red'):
    for col in cols:
        stat_playoffs_yes = df[df['playoffs_next_yr'] == 1]
        stat_playoffs_no = df[df['playoffs_next_yr'] == 0]
        n = len(cols)
    ncols = 3  # number of columns in grid
    nrows = int(np.ceil(n / ncols))  # rows needed

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()  # flatten 2D array to 1D for easy indexing

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.hist(stat_playoffs_yes[col], bins=12, alpha=0.7, label='Made Playoffs', color=yes_color)
        ax.hist(stat_playoffs_no[col], bins=12, alpha=0.7, label='Missed Playoffs', color=no_color)
        ax.set_title(f"Translation of {col} to Playoffs Next Year")
        ax.legend()

    # Turn off any extra empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def labeled_2d_scatter(df, cols, width=8, height=6, yes_color='green', no_color='red', alpha = 0.5):
    """
    Generate labeled scatter plots for all unique pairs of columns in `cols`.
    Arranges plots in a grid.
    """

    col_pairs = list(itertools.combinations(cols, 2))
    n = len(col_pairs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width * ncols, height * nrows))
    axes = axes.flatten()

    colors = df['playoffs_next_yr'].map({0: no_color, 1: yes_color})
    legend_elements = [
        Patch(facecolor=yes_color, label='Made Playoffs Next Year'),
        Patch(facecolor=no_color, label='Missed Playoffs Next Year')
    ]

    for i, (col1, col2) in enumerate(col_pairs):
        ax = axes[i]
        ax.scatter(df[col1], df[col2], c=colors, alpha=alpha, s=30)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f"{col1} vs {col2}")
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # Turn off any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def threshold_tuning_plot(y_test, y_prob, thresholds = np.arange(0.05, 0.95, 0.01)):
        precisions = []
        recalls = []
        f1s = []
        weighted_f1s = []   
        thresholds = thresholds

        for p in thresholds:
            y_pred = (y_prob >= p).astype(int)
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
            weighted_f1s.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))


        # Plotting all metrics
        plt.figure(figsize=(10,6));
        plt.plot(thresholds, precisions, label='Precision')
        plt.plot(thresholds, recalls, label='Recall')
        plt.plot(thresholds, f1s, label='F1 Score')
        plt.plot(thresholds, weighted_f1s, label='Weighted F1', linestyle='--')
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Metric vs. Classification Threshold")
        plt.legend()
        plt.grid(True)
        plt.show();