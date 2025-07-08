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

# 4. Weighted Logistic Regression Model

Due to the imbalance between teams who miss the playoffs (~60%) and teams who make the playoffs (~40%) each year, we fit a weighted logistic regression model. After reducing our feature set to avoid collinearity, recursive feature selection chooses point differential (PD), average experience (Avg Experience), and strength of schedule (SoS). However, the $p$-statistic associated with SoS when producing a three-feature model shows it is not significant:

### Logistic Regression Model Output

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


When used to make predictions on the test data, the model produces the following classification report which indicates a raw accuracy of 64.2%.

