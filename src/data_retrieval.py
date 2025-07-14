import pandas as pd
import random
import time
import requests
from bs4 import BeautifulSoup, Comment
from io import StringIO

'''Define dictionary which converts PFR team abbreviations to my preferred abbreviations for the corresponding franchise.'''
pfr_to_standard_tm = {
    # AFC East
    'buf': 'BUF',
    'nyj': 'NYJ',
    'mia': 'MIA',
    'nwe': 'NE',

    # AFC West
    'kan': 'KC',
    'lac': 'LAC',  # Los Angeles Chargers
    'sdg': 'LAC',  # San Diego Chargers (before 2017)
    'den': 'DEN',
    'rai': 'LV',   # Raiders (all of Oakland, LA, and Las Vegas location changes)

    # AFC North
    'pit': 'PIT',
    'rav': 'BAL',
    'cin': 'CIN',
    'cle': 'CLE', #1990-1995; 1999-.

    # AFC South
    'htx': 'HOU',  # Houston Texans (after 2002)
    'jax': 'JAX',
    'oti': 'TEN',  # Tennessee/Houston Oilers; Tennessee Titans
    'clt': 'IND',  

    # NFC East
    'phi': 'PHI',
    'nyg': 'NYG',
    'was': 'WAS',
    'dal': 'DAL',

    # NFC West
    'ram': 'LAR',  # Los Angeles/St. Louis Rams
    'sfo': 'SF',
    'sea': 'SEA',
    'crd': 'ARI',  # Arizona Cardinals

    # NFC North
    'gnb': 'GB',
    'det': 'DET',
    'min': 'MIN',
    'chi': 'CHI',

    # NFC South
    'tam': 'TB',
    'nor': 'NO',
    'car': 'CAR',
    'atl': 'ATL'
}


def scrape_pfr_rosters(year_start, year_end):
    '''Scrape yearly team rosters from PFR between year_start (inclusive) and year_end (exclusive). Includes exception case and time.sleep
    to avoid being banned by the site.'''

    team_abbrevs_pfr = ['buf', 'nyj', 'mia', 'nwe', 'kan', 'sdg', 'den', 'rai', 'pit', 'rav', 'cin', 'cle', 'htx', 'jax',
                    'oti', 'clt', 'phi', 'nyg', 'was', 'dal', 'ram', 'sfo', 'sea', 'crd', 'gnb', 'det', 'min', 'chi', 'tam', 'nor', 'car', 'atl']

    all_rosters = []
    for team in team_abbrevs_pfr:
        print(f"Starting team: {team.upper()}")
        for year in range(year_start, year_end):
            url = f"https://www.pro-football-reference.com/teams/{team}/{year}_roster.htm"
            print(f"  Processing year {year}...")

            try:
                res = requests.get(url)
                res.raise_for_status()
                soup = BeautifulSoup(res.content, 'html.parser')
                comments = soup.find_all(string=lambda text: isinstance(text, Comment))

                for comment in comments:
                    if 'id="roster"' in comment:
                        comment_soup = BeautifulSoup(comment, 'html.parser')
                        table = comment_soup.find('table', {'id': 'roster'})
                        if table:
                            df = pd.read_html(StringIO(str(table)))[0]

                            df['Year'] = year
                            df['Tm_raw'] = team
                            df['Tm'] = df['Tm_raw'].map(pfr_to_standard_tm)

                            all_rosters.append(df)
                            break

                time.sleep(random.uniform(6,10))

            except Exception as e:
                print(f"  Failed for {year}: {e}")

    df = pd.concat(all_rosters, ignore_index=True)
    return df


def load_seasonal_stats(path):
    '''Load seasonal stats from CSV and make small abbreviation fix due to some manual entry.'''

    df = pd.read_csv(path)
    df = df.replace('AZ', 'ARI')
    return df

def load_rosters(path):
    '''Load roster data from CSV. Specify columns to keep, then convert \"Rook\" to \"0\" in the experience column.nfl'''

    df = pd.read_csv(path)
    df = df[['Player', 'Year', 'Tm', 'Age', 'Pos', 'G', 'GS', 'Ht', 'Wt', 'College/Univ', 'BirthDate', 'Yrs', 'AV', 'Drafted (tm/rnd/yr)' ]]
    # Replacing 'Rook' with 0 in the 'Yrs' column
    df.loc[df["Yrs"] == "Rook", "Yrs"] = '0'
    return df