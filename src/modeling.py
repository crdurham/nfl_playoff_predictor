import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import \
(LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, precision_score,
                            recall_score, f1_score, roc_curve, auc, accuracy_score)
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
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)
    else:
        y = y.reindex(X.index)

    class_counts = Counter(y)
    n_total = len(y)

    sample_weights = y.map({
    0: n_total / (2 * class_counts[0]),
    1: n_total / (2 * class_counts[1])
    })

    model = sm.GLM(
        y,
        sm.add_constant(X),
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

def knn_cv(X, y, neighbors = 5, folds=10, repeats = 1, scoring='weighted_f1', verbose=True):
    scaler = StandardScaler()
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
    scores = []

    for i, (train_index, val_index) in enumerate(rskf.split(X, y), 1):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn.fit(X_train_scaled, y_train)      
        y_pred = knn.predict(X_val_scaled)

        if scoring =='f1':
            fold_score = f1_score(y_pred, y_val, average='binary')
        elif scoring == 'weighted_f1':
            fold_score = f1_score(y_pred, y_val, average='weighted')
        elif scoring == 'accuracy':
            fold_score = accuracy_score(y_pred, y_val)
        else:
            raise ValueError("Enter one of the following scoring metrics:\n" \
                             "'f1', 'weighted_f1, or 'accuracy'.")
        
        scores.append(fold_score)

        if verbose:
            print(f"Fold {i} {scoring}: {fold_score: .3f}")
    avg_score = np.mean(scores)
    if verbose:
        print(f"Average {scoring} across folds:\n {avg_score}")
    return scores, avg_score

def lda_cv(X, y, folds=10, repeats = 1, scoring='weighted_f1', verbose=True):
    scaler = StandardScaler()
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats,random_state=42)
    scores = []

    for i, (train_index, val_index) in enumerate(rskf.split(X, y), 1):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        lda = LDA()
        lda.fit(X_train_scaled, y_train)      
        y_pred = lda.predict(X_val_scaled)

        if scoring =='f1':
            fold_score = f1_score(y_pred, y_val, average='binary')
        elif scoring == 'weighted_f1':
            fold_score = f1_score(y_pred, y_val, average='weighted')
        elif scoring == 'accuracy':
            fold_score = accuracy_score(y_pred, y_val)
        else:
            raise ValueError("Enter one of the following scoring metrics:\n" \
                             "'f1', 'weighted_f1, or 'accuracy'.")
        
        scores.append(fold_score)

        if verbose:
            print(f"Fold {i} {scoring}: {fold_score: .3f}")
    avg_score = np.mean(scores)
    if verbose:
        print(f"Average {scoring} across folds:\n {avg_score}")
    return scores, avg_score

def lda_eval_threshold(X_train, X_test, y_train, y_test, threshold=0.5):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lda = LDA()
    lda.fit(X_train_scaled, y_train)

    y_prob = lda.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"Evaluation at threshold = {threshold:.2f}")
    print(classification_report(y_test, y_pred, digits=3))
    print(confusion_matrix(y_test, y_pred))

def lda_predict_probs(X_train, y_train, X_predict, features=['PD', 'OSRS'], threshold = 0.5):
    scaler = StandardScaler()
    lda = LDA()
    X_train_scaled = scaler.fit_transform(X_train[features])
    X_pred_scaled = scaler.transform(X_predict[features])
    lda.fit(X_train_scaled, y_train)
    y_probs = lda.predict_proba(X_pred_scaled)[:,1]
    y_pred = (y_probs >= threshold).astype(int)

    results = X_predict[['Tm']].copy()
    results['Playoff Prob'] = y_probs
    results['Make/Miss Playoffs'] = y_pred

    results = results.sort_values('Playoff Prob', ascending=False).reset_index(drop=True)

    return results

def glm_predict_probs(X_train, y_train, X_predict, features=['PD', 'OSRS'], threshold = 0.5):
    glm = fit_weighted_log(X_train[features], y_train)

    X_pred_with_const = sm.add_constant(X_predict[features])
    y_probs = glm.predict(X_pred_with_const)
    y_pred = (y_probs >= threshold).astype(int)

    results = X_predict[['Tm']].copy()
    results['Playoff Prob'] = y_probs
    results['Make/Miss Playoffs'] = y_pred

    results = results.sort_values('Playoff Prob', ascending=False).reset_index(drop=True)

    return results


def rf_cv(X, y, folds=10, repeats = 1, n_estimators = 100, max_depth=5, min_samples_leaf = 2, scoring='weighted_f1', verbose=True):
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats,random_state=42)
    scores = []

    for i, (train_index, val_index) in enumerate(rskf.split(X, y), 1):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     min_samples_leaf=min_samples_leaf, random_state=42)
        rf.fit(X_train, y_train)      
        y_pred = rf.predict(X_val)

        if scoring =='f1':
            fold_score = f1_score(y_pred, y_val, average='binary')
        elif scoring == 'weighted_f1':
            fold_score = f1_score(y_pred, y_val, average='weighted')
        elif scoring == 'accuracy':
            fold_score = accuracy_score(y_pred, y_val)
        else:
            raise ValueError("Enter one of the following scoring metrics:\n" \
                             "'f1', 'weighted_f1, or 'accuracy'.")
        
        scores.append(fold_score)

        if verbose:
            print(f"Fold {i} {scoring}: {fold_score: .3f}")
    avg_score = np.mean(scores)
    if verbose:
        print(f"Average {scoring} across folds:\n {avg_score}")
    return scores, avg_score


def KFold_statsmodels(X, y, k=5, threshold=0.61):

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    log_likelihoods = []
    accuracies = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        class_counts = Counter(y_train)
        n_total = len(y_train)
        w_train = y_train.map({
            0: n_total / (2 * class_counts[0]),
            1: n_total / (2 * class_counts[1])
        })

        X_train_sm = sm.add_constant(X_train)
        model = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial(), freq_weights=w_train)
        result = model.fit()

        X_val_sm = sm.add_constant(X_val)
        y_val_prob = result.predict(X_val_sm)

        y_val_pred = (y_val_prob >= threshold).astype(int)

        acc = (y_val_pred == y_val).mean()
        accuracies.append(acc)

        log_likelihoods.append(result.llf)

    print(f"Avg CV Accuracy at threshold {threshold}: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    print(f"Avg Train Log-Likelihood: {np.mean(log_likelihoods):.3f}")

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
        plt.show()

        best_idx = np.argmax(f1s)
        best_weighted_idx = np.argmax(weighted_f1s)
    
        return {
        'best_threshold_f1': thresholds[best_idx],
        'best_f1': f1s[best_idx],
        'best_threshold_weighted': thresholds[best_weighted_idx],
        'best_weighted_f1': weighted_f1s[best_weighted_idx]}


def CV_threshold_tuning_statsmodels(X, y, folds=5, thresholds=np.arange(0.05, 0.95, 0.01), plot=False):
    '''Performs k-fold cross validation manually using GLM in statsmodels as follows.
    (This description is as much for my own understanding as anything!)
    (1) Splits given X, y sets into train and val via StratifiedKFold. 
    (2) Sets class frequency weights for given fold (Counter).
    (3) Trains a model using X_train in GLM.
    (4) For each threshold: Makes predictions using X_val, evaluates performance on y_val.
                            Add F1 and weighted F1 to lists for fold.
    (5) Find best threshold (idx) for f1, wf1 for given fold.
    (6) Takes avg of thresholds, f1s/wf1s, across best from each fold.
    (7) (Optionally) Plot avg metric curves.'''
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    all_f1_scores = []
    all_weighted_f1_scores = []
    all_best_thresholds = []
    all_best_weighted_thresholds = []

    all_f1_scores_per_threshold = []
    all_weighted_f1_scores_per_threshold = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        class_counts = Counter(y_train)
        n_total = len(y_train)
        weights = y_train.map({
            0: n_total / (2 * class_counts[0]),
            1: n_total / (2 * class_counts[1])
        })

        # Train model
        X_train_sm = sm.add_constant(X_train)
        model = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial(), freq_weights=weights)
        result = model.fit()

        # Predict on val set
        X_val_sm = sm.add_constant(X_val)
        y_prob = result.predict(X_val_sm)

        # Tune threshold
        f1s = []
        weighted_f1s = []

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            weighted_f1s.append(f1_score(y_val, y_pred, average='weighted', zero_division=0))

        best_idx = np.argmax(f1s)
        best_weighted_idx = np.argmax(weighted_f1s)

        all_f1_scores.append(f1s[best_idx])
        all_weighted_f1_scores.append(weighted_f1s[best_weighted_idx])
        all_best_thresholds.append(thresholds[best_idx])
        all_best_weighted_thresholds.append(thresholds[best_weighted_idx])
        all_f1_scores_per_threshold.append(f1s)
        all_weighted_f1_scores_per_threshold.append(weighted_f1s)

        print(f"Fold {fold}: Best F1 = {f1s[best_idx]:.3f} at threshold {thresholds[best_idx]:.2f}, "
              f"Best Weighted F1 = {weighted_f1s[best_weighted_idx]:.3f} at threshold {thresholds[best_weighted_idx]:.2f}")

    
    if plot:
        plt.plot(thresholds, np.mean(all_f1_scores_per_threshold, axis=0), label='F1')
        plt.plot(thresholds, np.mean(all_weighted_f1_scores_per_threshold, axis=0), label='Weighted F1')
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Average CV Metrics vs Threshold")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Report averages
    #avg_f1 = np.mean(all_f1_scores)
    #avg_weighted_f1 = np.mean(all_weighted_f1_scores)
    #best_thresh = np.mean(all_best_thresholds)
    #best_weighted_thresh = np.mean(all_best_weighted_thresholds)
    avg_f1_curve = np.mean(all_f1_scores_per_threshold, axis=0)
    best_idx_curve = np.argmax(avg_f1_curve)
    best_thresh_curve = thresholds[best_idx_curve]
    best_f1_curve = avg_f1_curve[best_idx_curve]
    avg_weighted_f1_curve = np.mean(all_weighted_f1_scores_per_threshold, axis=0)
    best_weighted_idx_curve = np.argmax(avg_weighted_f1_curve)
    best_weighted_thresh_curve = thresholds[best_weighted_idx_curve]
    best_weighted_f1_curve = avg_weighted_f1_curve[best_weighted_idx_curve]

    print(f"\nBest F1 from averaged curve: {best_f1_curve:.3f} at threshold {best_thresh_curve:.2f}")
    print(f"Best Weighted F1 from averaged curve: {best_weighted_f1_curve:.3f} at threshold {best_weighted_thresh_curve:.2f}")
    #print(f"\nAverage best F1 across folds: {avg_f1:.3f} (mean threshold: {best_thresh:.2f})")
    #print(f"Average best Weighted F1 across folds: {avg_weighted_f1:.3f} (mean threshold: {best_weighted_thresh:.2f})")

    return {
        'best_thresh_curve': best_thresh_curve,
        'best_weighted_thresh_curve': best_weighted_thresh_curve,
        'peak_f1_of_avg': best_f1_curve,
        'peak_wf1_of_avg': best_weighted_f1_curve
        #'avg_best_f1': avg_f1,
        #'avg_best_weighted_f1': avg_weighted_f1,
        #'avg_threshold_f1': best_thresh,
        #'avg_threshold_weighted': best_weighted_thresh
    }


def CV_threshold_tuning_lda(X, y, folds=5, thresholds=np.arange(0.05, 0.95, 0.01), plot=False):
    '''Performs k-fold cross validation manually using LDA() in analogous fashion to CV_threshold_tuning_statsmodels.'''
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    all_f1_scores = []
    all_weighted_f1_scores = []
    all_best_thresholds = []
    all_best_weighted_thresholds = []

    all_f1_scores_per_threshold = []
    all_weighted_f1_scores_per_threshold = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        lda = LDA()
        lda.fit(X_train_scaled, y_train)

        y_prob = lda.predict_proba(X_val_scaled)[:,1]

        # Tune threshold
        f1s = []
        weighted_f1s = []

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            weighted_f1s.append(f1_score(y_val, y_pred, average='weighted', zero_division=0))

        best_idx = np.argmax(f1s)
        best_weighted_idx = np.argmax(weighted_f1s)

        all_f1_scores.append(f1s[best_idx])
        all_weighted_f1_scores.append(weighted_f1s[best_weighted_idx])
        all_best_thresholds.append(thresholds[best_idx])
        all_best_weighted_thresholds.append(thresholds[best_weighted_idx])
        all_f1_scores_per_threshold.append(f1s)
        all_weighted_f1_scores_per_threshold.append(weighted_f1s)

        print(f"Fold {fold}: Best F1 = {f1s[best_idx]:.3f} at threshold {thresholds[best_idx]:.2f}, "
              f"Best Weighted F1 = {weighted_f1s[best_weighted_idx]:.3f} at threshold {thresholds[best_weighted_idx]:.2f}")

    import matplotlib.pyplot as plt

    if plot:
        fig, ax = plt.subplots()
        ax.plot(thresholds, np.mean(all_f1_scores_per_threshold, axis=0), label='F1')
        ax.plot(thresholds, np.mean(all_weighted_f1_scores_per_threshold, axis=0), label='Weighted F1')
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Average CV Metrics vs Threshold")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        plt.show()

    avg_f1_curve = np.mean(all_f1_scores_per_threshold, axis=0)
    best_idx_curve = np.argmax(avg_f1_curve)
    best_thresh_curve = thresholds[best_idx_curve]
    best_f1_curve = avg_f1_curve[best_idx_curve]
    avg_weighted_f1_curve = np.mean(all_weighted_f1_scores_per_threshold, axis=0)
    best_weighted_idx_curve = np.argmax(avg_weighted_f1_curve)
    best_weighted_thresh_curve = thresholds[best_weighted_idx_curve]
    best_weighted_f1_curve = avg_weighted_f1_curve[best_weighted_idx_curve]

    print(f"\nBest F1 from averaged curve: {best_f1_curve:.3f} at threshold {best_thresh_curve:.2f}")
    print(f"Best Weighted F1 from averaged curve: {best_weighted_f1_curve:.3f} at threshold {best_weighted_thresh_curve:.2f}")

    results = {
        'best_thresh_curve': best_thresh_curve,
        'best_weighted_thresh_curve': best_weighted_thresh_curve,
        'peak_f1_of_avg': best_f1_curve,
        'peak_wf1_of_avg': best_weighted_f1_curve
    }

    if plot:
        return results, fig
    else:
        return results