import pandas as pd

def numeric_col(df, columns):
    for column in columns:
        df.loc[:, column] = pd.to_numeric(df[column], errors='coerce')
    return df

def basic_roster_info(df, year_start, year_end, min_games):
    avg_exp_age_data = []
    for year in range(year_start, year_end):
        rosters_in_year = df[df['Year'] == year]
        for team in rosters_in_year['Tm'].unique():
            team_roster = rosters_in_year[rosters_in_year['Tm'] == team]
            rel_roster = team_roster[team_roster['G'] >= min_games] #Restrict to players participating in at least min_games
            avg_exp = rel_roster['Yrs'].mean()
            avg_age = rel_roster['Age'].mean()
            num_rookies = len(team_roster[team_roster['Yrs'] == 0]) #Adding in ALL rookies
            avg_exp_age_data.append({
                'Year': year,
                'Tm': team,
                'Avg Age': avg_age,
                'Avg Experience': avg_exp,
                'Num Rookies': num_rookies
            })
    return pd.DataFrame(avg_exp_age_data)

def merge_roster_data(df_1, df_2):
    df = pd.merge(left=df_1,
                 right=df_2,
                 how='left',
                 left_on=['Tm', 'Season'],
                 right_on=['Tm', 'Year'])
    df = df.drop('Season', axis=1)
    df = df[['Year', 'Tm', 'W', 'L', 'win_percent', 'PF', 'PA', 'PD', 'MoV', 'SoS', 'SRS',
              'OSRS', 'DSRS', 'Avg Age', 'Avg Experience', 'Num Rookies', 'Playoffs']]
    return df

def sorted_playoffs_next(df):
    sorted_df = df.sort_values(by = ['Tm', 'Year']) #Group by team, then sort by year
    sorted_df['playoffs_next_yr'] = sorted_df.groupby('Tm')['Playoffs'].shift(-1) #Add shifted playoffs column; introduces NaN in some places
    sorted_df = sorted_df.dropna()
    return sorted_df