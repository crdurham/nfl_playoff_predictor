# **Predicting NFL Playoff Teams Based on Previous Season Stats**

## 1. Introduction

In the NFL, building sustained success is the goal of every franchise. However, given that roster changes due to any number of factors (free agency, trade, draft, injury) are a given each offseason, it is natural to ask what team-level statistics from the previous season can be used to predict team performance in the next season. In particular, we aim to predict whether a team will make or miss the playoffs as a broad indicator of team success. This is of interest to average fans, scouts, and of course bettors.

## 2. Data Sources

All data is obtained from Pro Football Reference. The main tables of interest are located at pages of the form https://www.pro-football-reference.com/years/2024/, where the year was allowed to range from 1990-2024. These tables contain basic annual statistics such as win percentage, points for/against, strength of schedule, and others.

For information relating to the roster constitution of each team, yearly rosters are available at pages of the form https://www.pro-football-reference.com/teams/buf/2024_roster.htm. Many of the columns are not relevant to our goals, so for each (team, season) pair we only retain the average age and experience of players who played in at least 8 games, as well as the number of rookies.

## 3. Exploratory Data Analysis

Simply put, our naive hope is that one or more variables will separate our data into two classes according to whether or not the team made the playoffs in the subsequent season. The plots below give a glimpse: the classes overlap substantially in 1 and 2 dimensions, as expected.

<p>
  <img src="https://github.com/crdurham/nfl_playoff_predictor/blob/main/images/histograms.png" width="600" height="400">
</p>
<p>
  <img src="https://github.com/crdurham/nfl_playoff_predictor/blob/main/images/2dscatters.png" width="800" height="800">

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

Through cross-validation, it is determined that the F1 score achieves a maximum value of $0.6$ at a classification threshold of $p=0.41$ while the weighted F1 score achieves a maximum value of $0.65$ at a classification threshold of $p=0.55$. This makes sense, since weighted F1 places greater emphasis on the majority class, i.e. teams which miss the playoffs. We elect to use the higher threshold $p=0.55$ for balance. The model generalizes to unseen test data as there is minimal change in performance.

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

We train an LDA model to compare with the linear regression model discussed above. We see as before that model performance with a more expansive feature set (SoS, Avg Experience, DSRS, Num Rookies in addition to PD and OSRS) is roughly the same as with only PD and OSRS. The sparse feature set yields an average weighted F1 score of $0.66$ in cross-validation, while on the test data

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

## 7. 2025 Predictions 

LDA and regression models output the same ordering of the teams in terms of probability to make the playoffs. The regression output it shown below.

```text
     Tm  Playoff Prob  Make/Miss Playoffs
0   DET      0.827910                   1
1   BAL      0.754216                   1
2   BUF      0.752531                   1
3   PHI      0.712961                   1
4    TB      0.695587                   1
5    GB      0.690405                   1
6   WAS      0.659058                   1
7   DEN      0.653443                   1
8   MIN      0.644396                   1
9   LAC      0.622421                   1
10  CIN      0.593345                   1
11   KC      0.561978                   1
12  ARI      0.525125                   0
13  PIT      0.523437                   0
14  SEA      0.494812                   0
15  HOU      0.472305                   0
16  LAR      0.458332                   0
17  ATL      0.446107                   0
18   SF      0.439805                   0
19  MIA      0.425909                   0
20  IND      0.420493                   0
21   NO      0.374640                   0
22  NYJ      0.373840                   0
23  CHI      0.361981                   0
24  DAL      0.317985                   0
25  JAX      0.305702                   0
26   LV      0.297814                   0
27   NE      0.273421                   0
28  TEN      0.265710                   0
29  NYG      0.248292                   0
30  CAR      0.241248                   0
31  CLE      0.219558                   0
```

## 7. Conclusions and Future Models

In broad strokes, these models match our intuition as football fans: a large point differential indicates a dominant team, which is an indicator of future success. Similarly, offensive strength has some level of continuity between seasons, likely due to the presence or lack of steady quarterback play.

One glaring omission from this work is any *nonlinearity*. This was intentional; it is crucial to test different models and different featues thoroughly, and this is only a small first step. We also neglected to include any roster continuity measurement, something which will certainly impact sustained success. 