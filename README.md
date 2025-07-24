# **Predicting NFL Playoff Teams Based on Previous Season Stats**

## 1. Introduction

In the NFL, building sustained success is the goal of every franchise. However, given that roster changes due to any number of factors (free agency, trade, draft, injury) are a given each offseason, it is natural to ask what team-level statistics from the previous season can be used to predict team performance in the next season. In particular, we aim to predict whether a team will make or miss the playoffs as a broad indicator of team success. This is of interest to average fans, scouts, and of course bettors.

## 2. Data Sources

All data is obtained from Pro Football Reference. The main tables of interest are located at pages of the form https://www.pro-football-reference.com/years/2024/, where the year was allowed to range from 1990-2024. These tables contain basic annual statistics such as win percentage, points for/against, strength of schedule, and others.

For information relating to the roster constitution of each team, yearly rosters are available at pages of the form https://www.pro-football-reference.com/teams/buf/2024_roster.htm. Many of the columns are not relevant to our goals, so for each (team, season) pair we only retain the average age and experience of players who played in at least 8 games, as well as the number of rookies.

## 3. Exploratory Data Analysis

Simply put, our naive hope is that one or more variables will separate our data into two classes according to whether or not the team made the playoffs in the subsequent season. The plots below give a glimpse: the classes overlap substantially in 1 and 2 dimensions, as expected.

<p>
  <img src="https://github.com/crdurham/nfl_playoff_predictor/blob/main/images/histograms.png" width="300" height="300">
  <img src="https://github.com/crdurham/nfl_playoff_predictor/blob/main/images/2dscatters.png" width="300" height="300">

</p>

## 4. Weighted Logistic Regression Model

### Feature Selection

Due to the imbalance between teams who miss the playoffs (~60%) and teams who make the playoffs (~40%) each year, we fit a weighted logistic regression model. After reducing our feature set to avoid collinearity, feature selection methods choose point differential (PD), and offensive simple rating system (OSRS) as the main features.

```text
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:       playoffs_next_yr   No. Observations:                  845
Model:                            GLM   Df Residuals:                   843.00
Model Family:                Binomial   Df Model:                            1
Link Function:                  Logit   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -542.76
Date:                Thu, 24 Jul 2025   Deviance:                       1085.5
Time:                        13:13:03   Pearson chi2:                     846.
No. Iterations:                     4   Pseudo R-squ. (CS):            0.09666
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
PD             0.0046      0.001      3.558      0.000       0.002       0.007
OSRS           0.0626      0.030      2.055      0.040       0.003       0.122
==============================================================================
```

Through cross-validation, it is determined that the F1 score achieves a maximum value of $0.6$ at a classification threshold of $p=0.41$ while the weighted F1 score achieves a maximum value of $0.65$ at a classification threshold of $p=0.55$. This makes sense, since weighted F1 places greater emphasis on the majority class, i.e. teams which miss the playoffs. The model generalizes to unseen test data as there is minimal change in performance.

```text
===============================================================
             precision    recall  f1-score   support
===============================================================
         0.0      0.714     0.742     0.728       128
===============================================================
         1.0      0.582     0.548     0.564        84
===============================================================
    accuracy                          0.665       212
===============================================================
    macro avg      0.648     0.645     0.646       212
===============================================================
  weighted avg      0.662     0.665     0.663       212
===============================================================
```

## 6. Linear Discriminant Analysis

We train an LDA model to compare with the linear regression model discussed above. Cross-validation reveals as before that model performance with a more expansive feature set (SoS, Avg Experience, DSRS, Num Rookies in addition to PD and OSRS) is roughly the same as with only PD and OSRS. The sparse feature set yielded an average weighted F1 score of $0.66$ in cross-validation, while on the test data

```text
===============================================================
             precision    recall  f1-score   support
===============================================================
         0.0       0.69      0.85      0.76       128
===============================================================
         1.0       0.65      0.43      0.52        84
===============================================================
    accuracy                           0.68       212
===============================================================
   macro avg       0.67      0.64      0.64       212
===============================================================
weighted avg       0.68      0.68      0.67       212
===============================================================
```
which indicates a similar level of generalizability and performance to the logistic regression model. It should be noted that the classification threshold was not tuned here.

## 7. Conclusions and Future Models

In broad strokes, these models matche our intuition as football fans: a large point differential indicates a dominant team, which is an indicator of future success. Similarly, offensive strength has some level of continuity between seasons, likely due to the presence or lack of steady quarterback play.

One glaring omission from this work is any *nonlinearity*. This was intentional; it is crucial to test different models and different featues thoroughly, and this is only a small first step. We also neglected to include any roster continuity measurement, something which will certainly impact sustained success. 