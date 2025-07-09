# **Predicting NFL Playoff Teams Based on Previous Season Stats**

## 1. Introduction

In the NFL, building sustained success is the goal of every franchise. However, given that roster changes due to any number of factors (free agency, trade, draft, injury) are a given each offseason, it is natural to ask what team-level statistics from the previous season can be used to predict team performance in the next season. In particular, we aim to predict whether a team will make or miss the playoffs as a broad indicator of team success. This is of interest to average fans, scouts, and of course bettors.

## 2. Data Sources

All data is obtained from Pro Football Reference. The main tables of interest are located at pages of the form https://www.pro-football-reference.com/years/2024/, where the year was allowed to range from 1990-2024. These tables contain basic annual statistics such as win percentage, points for/against, strength of schedule, and others.

For information relating to the roster constitution of each team, yearly rosters are available at pages of the form https://www.pro-football-reference.com/teams/buf/2024_roster.htm. Many of the columns are not relevant to our goals, so for each (team, season) pair we only retain the average age and experience of players who played in at least 8 games, as well as the number of rookies.

## 3. Exploratory Data Analysis

Simply put, our hope is that one or more variables will separate our data into two classes according to whether or not the team made the playoffs in the subsequent season. The plots below give a glimpse: no single feature will be able to distinguish the classes, as expected.

<p float="left">
  <img src="https://github.com/crdurham/nfl_playoff_predictor/blob/main/images/win_percent_to_playoffs.png" width="300" height="300">
  <img src="https://github.com/crdurham/nfl_playoff_predictor/blob/main/images/pd_to_playoffs.png" width="300" height="300">

</p>

## 4. Weighted Logistic Regression Model

Due to the imbalance between teams who miss the playoffs (~60%) and teams who make the playoffs (~40%) each year, we fit a weighted logistic regression model. After reducing our feature set to avoid collinearity, recursive feature selection chooses point differential (PD), average experience (Avg Experience), and strength of schedule (SoS). However, the $p$-statistic associated with SoS when producing a three-feature model shows it is not significant:

### Logistic Regression Model Output

```text
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:       playoffs_next_yr   No. Observations:                  845
Model:                            GLM   Df Residuals:                   841.00
Model Family:                Binomial   Df Model:                            3
Link Function:                  Logit   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -541.23
Date:                Tue, 10 Jun 2025   Deviance:                       1082.5
Time:                        21:30:53   Pearson chi2:                     841.
No. Iterations:                     4   Pseudo R-squ. (CS):            0.09993
Covariance Type:            nonrobust                                         
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
const              0.9301      0.463      2.007      0.045       0.022       1.838
PD                 0.0074      0.001      8.789      0.000       0.006       0.009
Avg Experience    -0.2586      0.121     -2.143      0.032      -0.495      -0.022
SoS                0.0752      0.049      1.549      0.121      -0.020       0.170
==================================================================================
```


For this reason as well as for interpretability, we fit a model which only uses PD and Avg Experience. From training, the model performs as follows:

```text
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:       playoffs_next_yr   No. Observations:                  845
Model:                            GLM   Df Residuals:                   842.00
Model Family:                Binomial   Df Model:                            2
Link Function:                  Logit   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -542.43
Date:                Tue, 10 Jun 2025   Deviance:                       1084.9
Time:                        21:30:44   Pearson chi2:                     841.
No. Iterations:                     4   Pseudo R-squ. (CS):            0.09736
Covariance Type:            nonrobust                                         
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
const              0.9121      0.462      1.974      0.048       0.007       1.818
PD                 0.0071      0.001      8.710      0.000       0.005       0.009
Avg Experience    -0.2534      0.120     -2.105      0.035      -0.489      -0.017
==================================================================================
```


When used to make predictions on the test data with a classification threshold of $p=0.5$, the model produces the following classification report which indicates a raw accuracy of 64.2%.

```text
              precision    recall  f1-score   support

         0.0      0.717     0.672     0.694       128
         1.0      0.543     0.595     0.568        84

    accuracy                          0.642       212
   macro avg      0.630     0.634     0.631       212
weighted avg      0.648     0.642     0.644       212
```

This accuracy improves upon the uninformed prediction that every team will miss the playoffs, which would achieve 60% accuracy. However, the recall and precision scores within the playoff teams (class 1) are a bit meager. We tune the classification parameter $p$ in an attempt to improve model performance.

<p float="left">
  <img src="/Users/coledurham/Documents/nfl_playoff_predictor/images/threshold_tuning.png" width="800" height="300">
</p>

The F1 score and weighted F1 score peak at $p=0.47$ and $p=0.54$ respectively. The former is more aggressive in predicting a team will make the playoffs because it values both classes equally and the recall for class 1 was low with $p=0.5$; the latter is a bit more conservative because it takes into account that the majority of teams fall into class 0. Something to keep in mind: which threshold we would use in practice ultimately depends on context. When scouting opponent strength, using the aggressive model makes sense.

We evaluate the model using each threshold specified above. First, at $p=0.47$

```text
        precision    recall  f1-score   support

0.0      0.759     0.641     0.695       128
1.0      0.558     0.690     0.617        84

accuracy                          0.660       212
macro avg      0.658     0.666    0.656       212
weighted avg   0.679     0.660    0.664       212
```

then at $p=0.54$

```text
        precision    recall  f1-score   support
0.0      0.720     0.742     0.731       128
1.0      0.588     0.560     0.573        84

accuracy                          0.670       212
macro avg      0.654     0.651     0.652       212
weighted avg   0.667     0.670     0.668       212
```

Both models perform at roughly the same accuracy overall on the training set. Upon performing 5-fold cross validation, the more aggressive model has an average accuracy of 60% while the more conservative model has an average accuracy of 64%.

## 5. Future Predictions

By using relevant statistics from the 2024 NFL season, we can estimate the likelihood that each team will make the playoffs in 2025.

| Tm   |   PD |   Avg Experience |   2025 Playoff Probability | 2025 Playoffs: Make or Miss   |
|------|------|------------------|----------------------------|-------------------------------|
| DET  |  222 |          4.10204 |                   0.809182 | Make                          |
| PHI  |  160 |          3.52083 |                   0.760052 | Make                          |
| GB   |  122 |          2.59184 |                   0.753853 | Make                          |
| BUF  |  157 |          3.96    |                   0.735062 | Make                          |
| BAL  |  157 |          4.08696 |                   0.728751 | Make                          |
| TB   |  117 |          3       |                   0.727199 | Make                          |
| DEN  |  114 |          3.12766 |                   0.71644  | Make                          |
| LAC  |  101 |          3.79245 |                   0.660696 | Make                          |
| CIN  |   38 |          2.84    |                   0.613401 | Make                          |
| MIN  |  100 |          4.73913 |                   0.603358 | Make                          |
| WAS  |   94 |          4.63265 |                   0.599641 | Make                          |
| KC   |   59 |          3.86667 |                   0.58667  | Make                          |
| ARI  |   21 |          3.19149 |                   0.562721 | Make                          |
| LAR  |  -19 |          2.38298 |                   0.543348 | Make                          |
| SEA  |    7 |          3.36    |                   0.52757  | Miss                          |
| PIT  |   33 |          4.52    |                   0.50015  | Miss                          |
| HOU  |    0 |          4.37255 |                   0.451231 | Miss                          |
| IND  |  -50 |          3.54717 |                   0.41566  | Miss                          |
| SF   |  -47 |          4       |                   0.393148 | Miss                          |
| ATL  |  -34 |          4.42857 |                   0.389211 | Miss                          |
| NO   |  -60 |          3.70588 |                   0.388973 | Miss                          |
| MIA  |  -19 |          4.97959 |                   0.381303 | Miss                          |
| CHI  |  -60 |          3.86    |                   0.379734 | Miss                          |
| NYJ  |  -66 |          4.40816 |                   0.338041 | Miss                          |
| LV   | -125 |          2.81132 |                   0.335096 | Miss                          |
| JAX  | -115 |          3.19231 |                   0.32939  | Miss                          |
| NE   | -128 |          3.06122 |                   0.316527 | Miss                          |
| DAL  | -118 |          3.8     |                   0.291906 | Miss                          |
| NYG  | -142 |          3.62    |                   0.266887 | Miss                          |
| TEN  | -149 |          3.71429 |                   0.252764 | Miss                          |
| CAR  | -193 |          3.94    |                   0.189594 | Miss                          |
| CLE  | -177 |          4.54902 |                   0.183373 | Miss                          |


According to our model, the five teams most likely to make the playoffs are the Detroit Lions, Philadelphia Eagles, Green Bay Packers, Buffalo Bills, and Baltimore Ravens. The five teams least likely to make the playoffs are the Dallas Cowboys, New York Giants, Tennessee Titans, Carolina Panthers, and Cleveland Browns. Some surprises here: the Chiefs were only given a 59% chance to make the playoffs, largely due to their low point differential last season; the Rams are more likely to make the playoffs than Seattle, Pittsburgh, and Houston despite having a significantly lower point differential (a consequence of a very young roster).

## 6. Conclusions and Future Models

In broad strokes, this model matches our intuition as football fans: a large point differential indicates a dominant team, which is an indicator of future success. Similarly, a young roster can be a source of potential or portend future improvement.

One glaring omission from this work is any *nonlinearity*. This was intentional; it is crucial to test different models and different featues thoroughly, and this is only a small first step. We also neglected to include any roster continuity measurement, something which will certainly impact sustained success. 