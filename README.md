# **Predicting NFL Playoff Teams Based on Previous Season Stats**

#1. Introduction

In the NFL, building sustained success is the goal of every franchise. However, given that roster changes due to any number of factors (free agency, trade, draft, injury) are a given each offseason, it is natural to ask what team-level statistics from the previous season can be used to predict team performance in the next season. In particular, we aim to predict whether a team will make or miss the playoffs as a broad indicator of team success. This is of interest to average fans, scouts, and of course bettors.

#2. Data Sources

All data is obtained from Pro Football Reference. The main tables of interest are located at pages of the form https://www.pro-football-reference.com/years/2024/, where the year was allowed to range from 1990-2024. These tables contain basic annual statistics such as win percentage, points for/against, strength of schedule, and others.

For information relating to the roster constitution of each team, yearly rosters are available at pages of the form https://www.pro-football-reference.com/teams/buf/2024_roster.htm. Many of the columns are not relevant to our goals, so for each (team, season) pair we only retain the average age and experience of players who played in at least 8 games, as well as the number of rookies.

#3. Exploratory Data Analysis

Simply put, our hope is that one or more variables will separate our data into two classes according to whether or not the team made the playoffs in the subsequent season.

<p align="left">
<img src="https://github.com/crdurham/nfl_playoff_predictor/blob/main/images/win_percent_to_playoffs.png" width="400" height="300">
</p>

<p align="right">
<img src="https://github.com/crdurham/nfl_playoff_predictor/blob/main/images/pd_to_playoffs.png" width="400" height="300">
</p>

